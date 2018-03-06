# author: Hao-ran Wei
# e-mail: whr94621@gmail.com

import theano
from theano import tensor as T
from theano.ifelse import ifelse
import numpy


from theanolayer.initializers import *
from theanolayer import nnet
from theanolayer.layers import dense, categorical_crossentropy
from theanolayer.layers.rnn_new import bidirectional_gru_layer
from theanolayer.utils import scope
from theanolayer.initializers import UniformInitializer

from src.utils import Vocab, GlobalNames
from src.models.base import Model
from .nn.layers import cgru_layer
from src.models.batch_beam_search import *


class NMTModel(Model):

    def __init__(self,
                 parameters,
                 n_words_src,
                 n_words_tgt,
                 dim,
                 dim_word,
                 enc_transition_depth=1,
                 dec_transition_depth=1,
                 dropout_rate=0.5,
                 tied_embedding=False,
                 Wemb=None,
                 Wemb_dec=None,
                 prefix="nmt"
                 ):

        super(NMTModel, self).__init__(parameters, prefix)

        self.n_words_src = n_words_src
        self.n_words_tgt = n_words_tgt
        self.dim = dim
        self.dim_word = dim_word
        self.enc_transition_depth = enc_transition_depth
        self.dec_transition_depth = dec_transition_depth
        self.dropout_rate = dropout_rate
        self.tied_embedding = tied_embedding

        self._Wemb = Wemb
        self._Wemb_dec = Wemb_dec

    @property
    def Wemb(self):
        if self._Wemb is not None:
            return self._Wemb
        else:
            Wemb = self.parameters.get_shared(name="Wemb",
                                              value_or_shape=(self.n_words_src, self.dim_word),
                                              initializer=UniformInitializer()
                                              )

            return Wemb

    @property
    def Wemb_dec(self):
        if self._Wemb_dec is not None:
            return self._Wemb_dec
        else:
            Wemb_dec = self.parameters.get_shared(name="Wemb_dec",
                                                  value_or_shape=(self.n_words_tgt, self.dim_word),
                                                  initializer=UniformInitializer()
                                                    )

            return Wemb_dec

    def set_embedding(self, Wemb=None, Wemb_dec=None):

        self._Wemb = Wemb
        self._Wemb_dec = Wemb_dec

    def _build_encoder(self, x, x_mask=None):

        encoder_scope = lambda name: self.model_scope(scope("encoder", name=name))

        emb = nnet.lookup(x, self.Wemb)

        res_fwd, res_bwd = bidirectional_gru_layer(parameters=self.parameters,
                                                   input=emb,
                                                   input_size=self.dim_word,
                                                   num_units=self.dim,
                                                   input_mask=x_mask,
                                                   transition_depth=self.enc_transition_depth,
                                                   prefix=encoder_scope("bigru"))

        ctx = T.concatenate([res_fwd, res_bwd], axis=res_fwd.ndim - 1)

        batch_size = x.shape[1]

        init_state = T.zeros(shape=(batch_size, self.dim), dtype='float32')  # We use zero bridge here

        return ctx, init_state

    def _build_decoder(self, y, ctx, init_state, Wemb_dec,
                       y_mask=None,
                       ctx_mask=None,
                       one_step=False,
                       ):

        decoder_scope = lambda name: self.model_scope(scope("decoder", name=name))

        """
        :type parameters: Parameters
        """

        dim_ctx = self.dim * 2

        if one_step is False:
            batch_size = y.shape[1]
            y_inp = T.concatenate([T.ones([1, batch_size], dtype='int64') * Vocab.BOS, y[:-1]], axis=0)
        else:
            y_inp = y

        emb = nnet.lookup(y_inp, Wemb_dec)

        proj = cgru_layer(parameters=self.parameters,
                          input=emb,
                          context=ctx,
                          input_size=self.dim_word,
                          num_units=self.dim,
                          context_size=dim_ctx,
                          input_mask=y_mask,
                          context_mask=ctx_mask,
                          init_state=init_state,
                          one_step=one_step,
                          prefix=decoder_scope('cgru'),
                          layer_normalization=False,
                          transition_depth=self.dec_transition_depth
                          )

        # hidden states of the decoder gru
        proj_h = proj[0]

        # weighted averages of context, generated by attention module
        ctxs = proj[1]

        # compute word probabilities

        preact_hidden = dense(parameters=self.parameters, inputs=proj_h,
                              input_size=self.dim,
                              num_units=self.dim_word,
                              prefix=decoder_scope("ff_preact_hidden"))

        preact_ctx = dense(parameters=self.parameters, inputs=ctxs,
                           input_size=dim_ctx,
                           num_units=self.dim_word,
                           prefix=decoder_scope("ff_preact_ctx"))

        preact_input = dense(parameters=self.parameters, inputs=emb,
                             input_size=self.dim_word,
                             num_units=self.dim_word,
                             prefix=decoder_scope("ff_preact_input"))

        preact = T.tanh(preact_hidden + preact_ctx + preact_input)

        if self.dropout_rate < 1.0:

            preact = nnet.dropout(input=preact, use_noise=GlobalNames.USE_NOISE,
                                  trng=GlobalNames.MY_DEFAULT_TRNG,
                                  keep_prob=self.dropout_rate)

        if self.tied_embedding is False:
            logit = dense(parameters=self.parameters, inputs=preact,
                          input_size=self.dim_word,
                          num_units=self.n_words_tgt,
                          prefix=decoder_scope("ff_logit"),
                          kernel_initializer=UniformInitializer()
                          )

        else:
            logit = T.dot(preact, Wemb_dec.T)

        return logit, proj, emb

    def build_model(self, x, x_mask, y, y_mask):

        ctx, init_state = self._build_encoder(x, x_mask)

        logit, *_ = \
            self._build_decoder(y, ctx, init_state, Wemb_dec=self.Wemb_dec,
            y_mask=y_mask, ctx_mask=x_mask, one_step=False)

        logit_shp = logit.shape
        probs = T.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1],
                                                   logit_shp[2]]))

        y_shp = y.shape

        cost = categorical_crossentropy(coding_dist=probs, true_dist=y.flatten(), mask=y_mask.flatten())
        cost = cost.reshape([y_shp[0], y_shp[1]])
        cost = cost.sum(0)

        return cost

    def _beam_search_step(self, time_, logits, prev_states, beam_scores, beam_mask, batch_size, beam_size):

        """
        :param logits: [batch_size * beam_size, vocab_size]

        :param acc_scores: [batch_size, beam_size]

        :param beam_mask: [batch_size, beam_size]

        :param next_states: [batch_size, beam_size, dim]
        """

        vocab_size = logits.shape[-1]

        # shape is [batch_size, beam_size, vocab_size]
        next_scores = - T.nnet.logsoftmax(logits)
        next_scores = next_scores.reshape([batch_size, beam_size, vocab_size])

        next_scores = mask_probs(scores=next_scores, beam_mask=beam_mask)

        # DISCUSS:
        # Whether add length penalty at each step
        next_beam_scores = next_scores + beam_scores[:, :, None]  # [batch_size, beam_size, vocab_size]

        next_beam_scores = next_beam_scores.reshape([batch_size, -1])  # [batch_size, beam_size * vocab_size]
        next_beam_scores = ifelse(T.eq(time_, 0.0), next_beam_scores[:, :vocab_size], next_beam_scores)

        # [batch_size, beam_size]
        next_beam_scores, word_indices = T.topk_and_argtopk(- next_beam_scores, axis=-1, sorted=False,
                                                                 kth=beam_size)
        next_beam_scores = - next_beam_scores

        # Final shape is [batch_size, beam_size]
        next_beam_ids, next_word_ids = T.divmod(word_indices, vocab_size)

        # shape is [batch_size, beam_size]
        next_beam_mask = tensor_gather_helper(gather_indices=next_beam_ids,
                                               gather_from=beam_mask,
                                               batch_size=batch_size,
                                               range_size=beam_size,
                                               gather_shape=[-1]
                                               )

        next_beam_mask_ = T.switch(T.eq(next_word_ids, numpy.int64(Vocab.EOS)), 0.0, 1.0)
        next_beam_mask = next_beam_mask * next_beam_mask_

        next_states = tensor_gather_helper(gather_indices=next_beam_ids,
                                            gather_from=prev_states,
                                            batch_size=batch_size,
                                            range_size=beam_size,
                                            gather_shape=[batch_size * beam_size, -1]
                                            )

        return next_beam_scores, next_beam_mask, next_beam_ids, next_word_ids, next_states

    def _build_batch_beam_search_decoder(self, ctx, init_state,
                                         beam_size=5, ctx_mask=None):

        batch_size = ctx.shape[1]

        def _step(time_, prev_word_ids, prev_states, beam_mask,
                  prev_beam_ids, prev_beam_scores,
                  Wemb_dec, ctx_mask, ctx  # non_sequence
                  ):
            """
            :param time: float32, number of time steps.

            :param prev_input: shape=[batch_size, beam_size]

            :param prev_states: shape=[batch_size, beam_size, dim]

            :param beam_mask: shape=[batch_size, beam_size]

            :param prev_beam_ids: shape=[batch_size, beam_size]

            :param prev_beam_scores: shape=[batch_size, beam_size]
            """

            batch_size = prev_word_ids.shape[0]
            beam_size = prev_word_ids.shape[1]

            prev_input_ = prev_word_ids.flatten()
            prev_states_ = prev_states.reshape([batch_size * beam_size, -1])
            beam_mask_ = beam_mask.flatten()

            logit, next_states, _ = \
                self._build_decoder(
                              y=prev_input_,
                              y_mask=beam_mask_,
                              init_state=prev_states_,
                              ctx=ctx,
                              ctx_mask=ctx_mask,
                              one_step=True,
                              Wemb_dec=Wemb_dec
                              )

            # [batch_size, beam_size, dim]
            next_states = next_states[0].reshape([batch_size, beam_size, -1])

            next_beam_scores, next_beam_mask, next_beam_ids, next_word_ids, next_states = \
                self._beam_search_step(time_=time_,
                                  logits=logit,
                                  prev_states=next_states,
                                  beam_scores=prev_beam_scores,
                                  beam_mask=beam_mask,
                                  batch_size=batch_size,
                                  beam_size=beam_size
                                  )

            return [time_ + 1.0, next_word_ids, next_states, next_beam_mask, next_beam_ids, next_beam_scores], {}, \
                   theano.scan_module.until(T.all(T.eq(next_beam_mask, 0.0)))

        init_time_ = T.constant(0.0, dtype='float32')
        init_word_ids = T.ones(shape=(batch_size, beam_size), dtype='int64') * Vocab.BOS

        init_states = T.tile(init_state[:, None, :], [1, beam_size, 1]).reshape([batch_size, beam_size, -1])
        init_beam_mask = T.ones(shape=[batch_size, beam_size], dtype='float32')
        init_beam_ids = T.zeros(shape=[batch_size, beam_size], dtype='int64')
        init_beam_scores = T.zeros(shape=[batch_size, beam_size], dtype='float32')

        ctx_ = T.tile(ctx[:, :, None, :], [1, 1, beam_size, 1]).reshape([ctx.shape[0], -1, ctx.shape[-1]])
        ctx_mask_ = T.tile(ctx_mask[:, :, None], [1, 1, beam_size]).reshape(
            [ctx_mask.shape[0], batch_size * beam_size])

        res, updates = theano.scan(_step,
                                   outputs_info=[init_time_, init_word_ids, init_states, init_beam_mask, init_beam_ids,
                                                 init_beam_scores],
                                   non_sequences=[self.Wemb_dec, ctx_mask_, ctx_],
                                   strict=False,
                                   n_steps=200
                                   )
        final_word_ids = res[1]  # [seq_len, batch_size, beam_size]
        final_beam_ids = res[-2]  # [seq_len, batch_size, beam_size]
        final_beam_scores = res[-1][-1]  # [batch_size, beam_size]

        return final_word_ids, final_beam_ids, final_beam_scores, updates

    def build_batch_beam_search(self, x, x_mask):

        ctx, init_state = \
            self._build_encoder(x=x,x_mask=x_mask)

        # [batch_size * beam_size, max_seq_len]
        batch_token_ids, batch_beam_ids, batch_scores, updates = \
            self._build_batch_beam_search_decoder(ctx=ctx, ctx_mask=x_mask, init_state=init_state)

        f_batch_beam_search = theano.function(inputs=[x, x_mask],
                                              outputs=[batch_token_ids, batch_beam_ids, batch_scores],
                                              updates=updates,
                                              name="f_batch_beam_search"
                                              )

        return f_batch_beam_search

    def _build_greedy_search_decoder(self, ctx, init_state, ctx_mask, Wemb_dec):

        batch_size = ctx.shape[1]

        init_word_ids = T.ones([batch_size, ], dtype='int64') * Vocab.BOS
        init_beam_mask = T.ones([batch_size, ], dtype='float32')

        def _step(prev_word_ids, prev_states, beam_mask, Wemb_dec, ctx_mask, ctx):

            logit, next_states, _ = \
                self._build_decoder(y=prev_word_ids,
                                    ctx=ctx,
                                    ctx_mask=ctx_mask,
                                    one_step=True,
                                    Wemb_dec=Wemb_dec,
                                    y_mask=beam_mask,
                                    init_state=prev_states)

            scores = - T.nnet.logsoftmax(logit) # [batch_size, ndim]

            next_word_ids = T.argmin(scores, axis=1).flatten() # [batch_size, ]
            next_word_ids = T.switch(T.eq(beam_mask, 0.0), Vocab.EOS, next_word_ids)

            next_beam_mask = T.switch(T.eq(next_word_ids, numpy.int64(Vocab.EOS)), 0.0, 1.0)
            next_beam_mask = next_beam_mask * beam_mask

            return [next_word_ids, next_states[0], next_beam_mask], {}, theano.scan_module.until(T.all(T.eq(next_beam_mask, 0.0)))

        res, updates  = theano.scan(fn=_step,
                              outputs_info=[init_word_ids, init_state, init_beam_mask],
                              non_sequences=[Wemb_dec, ctx_mask, ctx],
                              n_steps=200,
                              strict=False
                              )
        final_word_ids = res[0] # [max_len, batch_size]
        final_word_ids = T.transpose(final_word_ids, [1, 0])

        return final_word_ids, updates

    def build_greedy_search(self, x, x_mask):

        ctx, init_state = \
            self._build_encoder(x=x, x_mask=x_mask)

        final_word_ids, updates = self._build_greedy_search_decoder(ctx, init_state, x_mask, self.Wemb_dec)

        f_greedy_decode = theano.function(inputs=[x, x_mask],
                                          outputs=final_word_ids,
                                          updates=updates,
                                          name="f_greedy_decode")

        return f_greedy_decode