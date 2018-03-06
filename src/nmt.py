import os
import json
import time
import numpy as np
import theano
from tensorboardX import SummaryWriter
import sys

from theano import tensor as T
from theanolayer.parameters import Parameters
from theanolayer.optimizers import *

from src.utils import *
from src.data_io import *
from src.models.batch_beam_search import process_beam_results
from src.models.models import NMTModel
from src.metric.bleu_scorer import ExternalScriptBLEUScorer

def prepare_data(seqs_x, seqs_y=None):

    def _np_pad_batch_2D(samples, pad):

        batch_size = len(samples)

        sizes = [len(s) for s in samples]
        max_size = max(sizes)

        x = np.full((batch_size, max_size), fill_value=pad, dtype='int32')

        for ii in range(batch_size):
            x[ii,:sizes[ii]] = samples[ii]

        x_sizes = np.array(sizes).astype('int64')

        return np.transpose(x, [1,0]), np.transpose(sequence_mask(x_sizes), [1,0])

    seqs_x = list(map(lambda s: s + [Vocab.EOS], seqs_x))
    x, x_mask = _np_pad_batch_2D(samples=seqs_x, pad=Vocab.EOS)

    if seqs_y is None:
        return x, x_mask

    seqs_y = list(map(lambda s: s + [Vocab.EOS], seqs_y))
    y, y_mask = _np_pad_batch_2D(seqs_y, pad=Vocab.EOS)

    return x, x_mask, y, y_mask

def build_training_loss_eval(parameters, model, optimizer_configs):

    """
    :type parameters: Parameters

    :type model: NMTModel
    """
    x = T.matrix(name="x", dtype='int64')
    x_mask = T.matrix(name="x_mask", dtype='float32')

    y = T.matrix(name="x", dtype='int64')
    y_mask = T.matrix(name="x_mask", dtype='float32')

    loss_batch = model.build_model(x, x_mask, y, y_mask)

    inps = [x, x_mask, y, y_mask]

    func_loss_eval = theano.function(inps, outputs=loss_batch)

    loss = loss_batch.mean()

    if optimizer_configs['weights_regularization'] > 0.0:
        weight_decay = apply_regularization(parameters=parameters,
                                            decay_c=optimizer_configs['weights_regularization'])

        loss += weight_decay

    f_cost, f_update = \
        build_optimizer(optimizer=optimizer_configs['optimizer'],
                        parameters=parameters,
                        loss=loss,
                        inputs=inps,
                        **optimizer_configs['optimizer_params']
                        )

    return f_cost, f_update, func_loss_eval

def loss_validation(f_loss_eval, valid_data_reader, prepare_data):
    """
    :type valid_data_reader: DataIterator
    """
    valid_iter = valid_data_reader.build_generator()
    loss_per_sent = []

    for seqs_x, seqs_y in valid_iter:

        x, x_mask, y, y_mask = prepare_data(seqs_x, seqs_y)

        loss_ = f_loss_eval(x, x_mask, y, y_mask)

        if np.any(np.isnan(loss_)):
            WARN("NaN detected!")

        for ll in loss_:
            loss_per_sent.append(ll)

    return float(np.mean(loss_per_sent))


def build_decoding_func(parameters, model):
    """
    :type parameters: Parameters

    :type model: NMTModel
    """

    Wemb = parameters.get_shared(name="Wemb")

    model.set_embedding(Wemb=Wemb)

    x = T.matrix(name="x", dtype='int64')
    x_mask = T.matrix(name="x_mask", dtype='float32')

    f_decode = model.build_batch_beam_search(x, x_mask)

    return f_decode

def bleu_validation(func_beam_search,
                    valid_data_reader,
                    prepare_data,
                    bleu_scorer,
                    vocab_tgt,
                    uidx,
                    batch_size=None,
                    tgt_bpe=None,
                    debug=False,
                    eval_at_char_level=False
                    ):

    """
    :type valid_data_reader: DataIterator

    :type vocab_tgt: Vocab

    :type bleu_scorer: ExternalScriptBLEUScorer
    """

    def _split_into_chars(line):
        new_line = []
        for w in line:
            if vocab_tgt.token2id(w) != 1:
                # if not UNK, split into characters
                new_line += list(w)
            else:
                # if UNK, treat as a special character
                new_line.append(w)

        return new_line

    trans = []

    valid_iter = valid_data_reader.build_generator(batch_size=batch_size)

    for n, (seqs_x, _) in enumerate(valid_iter):

        x, x_mask = prepare_data(seqs_x)

        token_ids_b, beam_ids_b, scores_b = func_beam_search(x, x_mask)

        for tt in range(scores_b.shape[0]):

            sent_t = process_beam_results(final_word_ids=token_ids_b[:,tt,:],
                                          final_beam_ids=beam_ids_b[:,tt,:],
                                          final_scores=scores_b[tt]
                                          )
            sent_t = sent_t.tolist()
            sent_t = [[wid for wid in line if wid != Vocab.EOS] + [Vocab.EOS] for line in sent_t]

            x_tokens = []
            for wid in sent_t[0]:
                if wid == Vocab.EOS:
                    break
                x_tokens.append(vocab_tgt.id2token(wid))

            trans.append(' '.join(x_tokens))

    # Merge bpe segmentation
    trans = [line.replace("@@ ", "") for line in trans]

    # Split into characters
    if eval_at_char_level is True:
        trans = [' '.join(_split_into_chars(line.strip().split())) for line in trans]

    if not os.path.exists("./valid"):
        os.mkdir("./valid")

    hyp_path = './valid/trans.iter{0}.txt'.format(uidx)

    with open(hyp_path, 'w') as f:
        for line in trans:
            f.write('%s\n' % line)

    bleu_v = bleu_scorer.corpus_bleu(hyp_path)

    return bleu_v


def train(FLAGS):
    """
    FLAGS:
        saveto: str
        reload: store_true
        cofig_path: str
        pretrain_path: str, defalut=""
        model_name: str
        log_path: str
    """
    config_path = os.path.abspath(FLAGS.config_path)

    with open(config_path.strip()) as f:
        configs = json.load(f)

    data_configs = configs['data_configs']
    # data_configs = set_default_configs(data_configs, default_configs['data_configs'])

    model_configs = configs['model_configs']
    optimizer_configs = configs['optimizer_configs']

    training_configs = configs['training_configs']
    # training_configs = set_default_configs(training_configs, default_configs['training_configs'])

    saveto_collections = '%s.pkl' % os.path.join(FLAGS.saveto, FLAGS.model_name + GlobalNames.MY_CHECKPOINIS_PREFIX)
    saveto_best_model = os.path.join(FLAGS.saveto, FLAGS.model_name + GlobalNames.MY_BEST_MODEL_SUFFIX)

    timer = Timer()

    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary

    vocab_src = Vocab(dict_path=data_configs['dictionaries'][0], max_n_words=data_configs['n_words_src'])
    vocab_tgt = Vocab(dict_path=data_configs['dictionaries'][1], max_n_words=data_configs['n_words_tgt'])


    train_bitext_dataset = ZipDatasets(
        TextDataset(data_path=data_configs['train_data'][0],
                    vocab=vocab_src,
                    bpe_codes=data_configs['src_bpe_codes'],
                    max_len=data_configs['max_len'][0]
                    ),
        TextDataset(data_path=data_configs['train_data'][1],
                    vocab=vocab_tgt,
                    bpe_codes=data_configs['tgt_bpe_codes'],
                    max_len=data_configs['max_len'][1]
                    ),
        shuffle=training_configs['shuffle']
    )

    valid_bitext_dataset = ZipDatasets(
        TextDataset(data_path=data_configs['valid_data'][0],
                    vocab=vocab_src,
                    bpe_codes=data_configs['src_bpe_codes']),
        TextDataset(data_path=data_configs['valid_data'][1],
                    vocab=vocab_tgt,
                    bpe_codes=data_configs['tgt_bpe_codes'])
    )

    train_reader = DataIterator(dataset=train_bitext_dataset,
                                batch_size=training_configs['batch_size'],
                                sort_buffer=True,
                                sort_fn=lambda line : len(line[1]))

    valid_reader = DataIterator(dataset=valid_bitext_dataset,
                                batch_size=training_configs['valid_batch_size'],
                                sort_buffer=False)

    # bleu_scorer = BLEUScorer(reference_path=['{0}{1}'.format(data_configs['valid_data'][1], i) for i in range(data_configs['n_refs'])],
    #                          use_char=training_configs["eval_at_char_level"])

    bleu_scorer = ExternalScriptBLEUScorer(reference_path=data_configs['bleu_valid_reference'],
                                           lang_pair=data_configs['lang_pair'])

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Reload theano shared variables

    parameters = Parameters()
    model_collections = Collections()

    if len(FLAGS.pretrain_path) > 0 and (FLAGS.reload is False or not os.path.exists(FLAGS.saveto)):
        if os.path.exists(FLAGS.pretrain_path):
            INFO('Reloading model parameters...')
            timer.tic()
            params_pretrain = np.load(FLAGS.pretrain_path)
            parameters.load(params=params_pretrain)
            INFO('Done. Elapsed time {0}'.format(timer.toc()))
        else:
            WARN("Pre-trained model not found at {0}!".format(FLAGS.pretrain_path))

    if FLAGS.reload is True and os.path.exists(saveto_best_model):
        INFO('Reloading model...')
        timer.tic()
        params = np.load(saveto_best_model)
        parameters.load(params)

        model_archives = Collections.unpickle(path=saveto_collections)
        model_collections.load(archives=model_archives)

        uidx = model_archives['uidx']
        bad_count = model_archives['bad_count']

        INFO('Done. Elapsed time {0}'.format(timer.toc()))

    else:
        uidx = 0
        bad_count = 0

    lrate = optimizer_configs['learning_rate']
    is_early_stop = False

    # ================================================================================== #
    # Build Model & Sampler & Validation
    INFO('Building model...')
    timer.tic()

    nmt_model = NMTModel(parameters=parameters,
                         prefix=FLAGS.model_name,
                         n_words_src=vocab_src.max_n_words,
                         n_words_tgt=vocab_tgt.max_n_words,
                         **model_configs)

    INFO('Building training&loss_eval function...')
    f_cost, f_update, f_loss_eval = \
        build_training_loss_eval(parameters=parameters,
                                 model=nmt_model,
                                 optimizer_configs=optimizer_configs
                                 )

    INFO('Building decoding function...')

    f_decoding = build_decoding_func(parameters, nmt_model)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    # ================================================================================== #
    # Prepare training

    params_best_loss = None

    summary_writer = SummaryWriter(log_dir=FLAGS.log_path)

    # if lrate_decay_patience=20, loss_valid_freq=100, lrate_scheduler start at 20 * 100 = 2000 steps
    lrate_scheduler = LearningRateDecay(max_patience=training_configs['lrate_decay_patience'],
                                        start_steps=training_configs['loss_valid_freq'] * training_configs['lrate_decay_patience'])

    cum_samples = 0
    cum_words = 0
    saving_files = []

    # Timer for computing speed
    timer_for_speed = Timer()
    timer_for_speed.tic()

    INFO('Begin training...')

    for eidx in range(training_configs['max_epochs']):
        summary_writer.add_scalar("Epoch", (eidx + 1), uidx)

        n_samples = 0

        train_iter = train_reader.build_generator()

        for seqs_x, seqs_y in train_iter:

            b_samples = len(seqs_y)
            b_words = sum([len(s) for s in seqs_y])

            n_samples += b_samples
            uidx += 1

            x, x_mask, y, y_mask = prepare_data(seqs_x, seqs_y)

            cum_samples += b_samples
            cum_words += b_words

            GlobalNames.USE_NOISE.set_value(1.0)

            ud_start = time.time()

            loss_ = f_cost(x, x_mask, y, y_mask)
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if np.isnan(loss_) or np.isinf(loss_):
                WARN('NaN detected')
                time.sleep(300) # if NaN is detected, sleep 5min

            # ================================================================================== #
            # Verbose
            if np.mod(uidx, training_configs['disp_freq']) == 0 or FLAGS.debug:

                INFO("Epoch: {0} Update: {1} Loss: {2:.2f} UD: {3:.2f} {4:.2f} words/sec {5:.2f} samples/sec".format(
                    eidx, uidx, float(loss_), ud,
                    cum_words / timer_for_speed.toc(return_seconds=True),
                    cum_samples / timer_for_speed.toc(return_seconds=True)
                ))
                cum_words = 0
                cum_samples = 0
                timer_for_speed.tic()

            # ================================================================================== #
            # Saving checkpoints

            if np.mod(uidx, training_configs['save_freq']) == 0 or FLAGS.debug:

                INFO('Saving the model at iteration {}...'.format(uidx))

                params_uidx = parameters.export()

                saveto_uidx = os.path.join(FLAGS.saveto, FLAGS.model_name + '.%d.npz' % uidx)

                np.savez(saveto_uidx, **params_uidx)

                Collections.pickle(path=saveto_collections,
                                   uidx=uidx,
                                   bad_count=bad_count,
                                   **model_collections.export())

                saving_files.append(saveto_uidx)

                INFO('Done')

                # ================================================================================== #
                # Remove models

                if len(saving_files) > 5:
                    for f in saving_files[:-1]:
                            os.remove(f)

                    saving_files = [saving_files[-1]]


            # ================================================================================== #
            # Loss Validation & Learning rate annealing

            if np.mod(uidx, training_configs['loss_valid_freq']) == 0 or FLAGS.debug:

                GlobalNames.USE_NOISE.set_value(0.0)

                valid_loss = loss_validation(f_loss_eval=f_loss_eval,
                                             valid_data_reader=valid_reader,
                                             prepare_data=prepare_data
                                             )

                model_collections.add_to_collection("history_losses", valid_loss)

                min_history_loss = np.array(model_collections.get_collection("history_losses")).min()

                summary_writer.add_scalar("loss", valid_loss, global_step=uidx)
                summary_writer.add_scalar("best_loss", min_history_loss, global_step=uidx)

                # If no bess loss model saved, save it.
                if len(model_collections.get_collection("history_losses")) == 0 or params_best_loss is None:
                    params_best_loss = parameters.export()

                if valid_loss <= min_history_loss:

                    params_best_loss = parameters.export() # Export best variables

                if training_configs['lrate_decay'] is True or FLAGS.debug:

                    new_lrate = lrate_scheduler.decay(uidx, valid_loss, lrate)

                    summary_writer.add_scalar("lrate_half_patience", lrate_scheduler._bad_counts, uidx)


                    # If learning rate decay happened,
                    # reload from the best loss model.
                    if new_lrate < lrate:
                        parameters.reload_value(params_best_loss, exclude=None)

                    lrate = new_lrate

                    summary_writer.add_scalar("lrate", lrate, uidx)

            # ================================================================================== #
            # BLEU Validation & Early Stop

            if np.mod(uidx, training_configs['bleu_valid_freq']) == 0 or FLAGS.debug:

                GlobalNames.USE_NOISE.set_value(0.0)

                valid_bleu = bleu_validation(func_beam_search=f_decoding,
                                             valid_data_reader=valid_reader,
                                             prepare_data=prepare_data,
                                             bleu_scorer=bleu_scorer,
                                             vocab_tgt=vocab_tgt,
                                             uidx=uidx,
                                             debug=FLAGS.debug,
                                             tgt_bpe=data_configs['tgt_bpe_codes'],
                                             batch_size=training_configs['bleu_valid_batch_size'],
                                             eval_at_char_level=training_configs["eval_at_char_level"]
                                             )

                model_collections.add_to_collection(key="history_bleus", value=valid_bleu)

                best_valid_bleu = float(np.array(model_collections.get_collection("history_bleus")).max())

                summary_writer.add_scalar("bleu", valid_bleu, uidx)
                summary_writer.add_scalar("best_bleu", best_valid_bleu, uidx)


                # If model get new best valid bleu score
                if valid_bleu >= best_valid_bleu:
                    bad_count = 0

                    if is_early_stop is False:
                        INFO('Saving best model...')

                        best_params = parameters.export()
                        np.savez(saveto_best_model,
                                 **best_params)

                        INFO('Done.')

                else:
                    bad_count += 1

                    if bad_count >= training_configs['early_stop_patience']:
                        is_early_stop = True
                        WARN("Early Stop!")

                summary_writer.add_scalar("bad_count", bad_count, uidx)

                with open("./valid.txt", 'a') as f:
                    f.write("{0} Loss: {1:.2f} BLEU: {2:.2f} lrate: {3:6f} patience: {4}\n".format(uidx, valid_loss, valid_bleu, lrate, bad_count))


def translate(FLAGS):

    config_path = os.path.abspath(FLAGS.config_path)

    with open(config_path.strip()) as f:
        configs = json.load(f)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']

    timer = Timer()
    # ================================================================================== #
    # Load Data

    INFO('Loading data...')
    timer.tic()

    # Generate target dictionary
    vocab_src = Vocab(dict_path=FLAGS.source_dict_path, max_n_words=data_configs['n_words_src'])
    vocab_tgt = Vocab(dict_path=FLAGS.target_dict_path, max_n_words=data_configs['n_words_tgt'])

    valid_dataset = TextDataset(data_path=FLAGS.source_path, vocab=vocab_src, bpe_codes=FLAGS.source_bpe_codes)

    source_reader = DataIterator(dataset=valid_dataset, batch_size=FLAGS.batch_size, sort_buffer=False)

    parameters = Parameters()

    INFO('Reloading model parameters...')
    timer.tic()

    params = np.load(FLAGS.model_path)
    parameters.load(params=params)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Building decoding function...')
    timer.tic()
    nmt_model = NMTModel(parameters=parameters,
                         prefix=FLAGS.model_name,
                         n_words_src=vocab_src.max_n_words,
                         n_words_tgt=vocab_tgt.max_n_words,
                         **model_configs)

    f_decoding = build_decoding_func(parameters, nmt_model)

    INFO('Done. Elapsed time {0}'.format(timer.toc()))

    INFO('Begin...')

    source_iter = source_reader.build_generator(batch_size=FLAGS.batch_size)

    n_done = 0
    n_words = 0

    result = []

    timer.tic()

    GlobalNames.USE_NOISE.set_value(0.0)

    for seqs_x in source_iter:

        x, x_mask  = prepare_data(seqs_x=seqs_x)

        token_ids_b, beam_ids_b, scores_b = f_decoding(x, x_mask)

        for tt in range(scores_b.shape[0]):

            sent_t = process_beam_results(final_word_ids=token_ids_b[:,tt,:],
                                          final_beam_ids=beam_ids_b[:,tt,:],
                                          final_scores=scores_b[tt]
                                          )
            sent_t = sent_t.tolist()
            sent_t = [[wid for wid in line if wid != Vocab.EOS] + [Vocab.EOS] for line in sent_t]

            n_words += len(sent_t[0])

            result.append(sent_t)

        n_done += len(seqs_x)
        INFO('Translated %d sentences' % n_done)

    INFO('Done. Speed: {0:.2f} words/sec'.format(n_words / (timer.toc(return_seconds=True))))

    translation = []
    for sent in result:
        samples = []
        for trans in sent:
            sample = []
            for w in trans:
                if w == 0:
                    break
                sample.append(vocab_tgt.id2token(w))
            samples.append(' '.join(sample))
        translation.append(samples)

    keep_n = FLAGS.beam_size if FLAGS.keep_n <= 0 else min(FLAGS.beam_size, FLAGS.keep_n)
    outputs = ['%s.%d' % (FLAGS.saveto, i) for i in range(keep_n)]

    with batch_open(outputs, 'w') as handles:
        for trans in translation:
            for i in range(FLAGS.beam_size):
                if i < len(trans):
                    handles[i].write('%s\n' % trans[i])
                else:
                    handles[i].write('%s\n' % 'eos')