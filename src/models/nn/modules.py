import theano.tensor as T
from theanolayer.nnet.base import Module

MINOR = 1e-7

class MLPAttention(Module):

    def __init__(self,
                 query_size,
                 mem_size,
                 parameters,
                 prefix="attn"):

        super(MLPAttention, self).__init__(parameters, prefix)

        self.query_size = query_size
        self.mem_size = mem_size

        self._register_params(name="W_att",
                              value_or_shape=(mem_size, mem_size))

        self._register_params(name="b_att",
                              value_or_shape=(mem_size, ))

        self._register_params(name="U_att",
                              value_or_shape=(query_size, mem_size))

        self._register_params(name="v_att",
                              value_or_shape=(mem_size, 1))

        self._register_params(name="c_att",
                              value_or_shape=(1, ))

    def precal_mem(self, mem):

        return T.dot(mem, self._get_shared("W_att")) + self._get_shared("b_att")

    def forward(self, precal_mem, query, mem, mem_mask=None, *args, **kwargs):

        preact_q = T.dot(query, self._get_shared("U_att"))

        preact = preact_q[None, :, :] + precal_mem

        preact = T.tanh(preact)

        logit = T.dot(preact, self._get_shared("v_att")) + self._get_shared("c_att")

        logit = logit.reshape([logit.shape[0], logit.shape[1]])

        attn_weights = T.exp(logit - logit.max(0, keepdims=True))

        if mem_mask is not None:
            attn_weights = attn_weights * mem_mask

        attn_weights = attn_weights / (attn_weights.sum(0, keepdims=True) + MINOR)

        attn_values = (mem * attn_weights[:,:,None]).sum(0)

        return attn_weights, attn_values