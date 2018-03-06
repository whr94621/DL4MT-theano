import theano
from theano import tensor as T

from theanolayer.utils import scope
from theanolayer.nnet.rnn_cell import GRUCell
from .modules import MLPAttention

MINOR = 1e-7

def cgru_layer(parameters,
               input,
               context,
               input_size,
               num_units,
               context_size,
               input_mask=None,
               context_mask=None,
               init_state=None,
               activation=T.tanh,
               one_step=False,
               prefix='cgru_layer',
               transition_depth=1,
               **kwargs
               ):

    if not callable(activation):
        raise ValueError("activation must be callable.")


    n_timesteps = input.shape[0]
    n_timesteps_ctx = context.shape[0]

    if input.ndim == 3:
        batch_size = input.shape[1]
    else:
        batch_size = 1

    if input_mask is None:
        input_mask = T.alloc(1., n_timesteps, batch_size)

    if context_mask is None:
        context_mask = T.alloc(1., n_timesteps_ctx, batch_size)

    if one_step is True:
        assert init_state is not None, 'previous state must be provided'

    if init_state is None:
        init_state = T.alloc(0., batch_size, num_units)

    gru_ctx = GRUCell(parameters=parameters,
                      input_size=input_size,
                      num_units=num_units, prefix=scope(prefix,"gru_ctx"))

    gru_hidden = GRUCell(parameters=parameters,
                      input_size=context_size,
                      num_units=num_units, prefix=scope(prefix, "gru_hidden"),
                      transition_depth=transition_depth)

    mlp_attn = MLPAttention(parameters=parameters,
                            query_size=num_units,
                            mem_size=context_size,
                            prefix=scope(prefix, "attn"))


    precal_x = gru_ctx.precal_x(input)
    precal_ctx = mlp_attn.precal_mem(context)


    def _step(mask_, precal_x,
              prev_h, attn_v, attn_w,
              precal_ctx, ctx, *args):

        args = list(args)

        gru_ctx_params = [None, None] + args[:gru_ctx.num_params - 2]
        attn_params = [None, None] + args[gru_ctx.num_params - 2:(gru_ctx.num_params + mlp_attn.num_params - 4)]
        gru_hidden_params = args[(gru_ctx.num_params + mlp_attn.num_params - 4):]

        h1 = gru_ctx(precal_x, prev_h, precal_x=True, params=gru_ctx_params)

        h1 = mask_[:, None] * h1 + (1. - mask_)[:, None] * prev_h

        attn_w, attn_v = mlp_attn(precal_ctx, h1, ctx, context_mask, params=attn_params)

        h2 = gru_hidden(attn_v, h1, params=gru_hidden_params)

        h2 = mask_[:, None] * h2 + (1. - mask_)[:, None] * prev_h

        return h2, attn_v, attn_w.T

    seqs = [input_mask, precal_x]

    outputs_info = [init_state, T.alloc(0., batch_size, context.shape[2]), T.alloc(0., batch_size, context.shape[0])]

    non_sequences = [precal_ctx, context] + \
        gru_ctx.pack_params()[2:] + \
        mlp_attn.pack_params()[2:] + \
        gru_hidden.pack_params()

    if one_step is True:
        res = _step(*(seqs + outputs_info + non_sequences))
    else:
        res, _ = theano.scan(fn=_step, sequences=seqs,
                             outputs_info=outputs_info,
                             non_sequences=non_sequences,
                             name=scope(prefix, "scan"),
                             n_steps=n_timesteps,
                             strict=False)
    return res