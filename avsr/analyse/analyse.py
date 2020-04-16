import matplotlib.pyplot as plt
import Levenshtein
import numpy as np


def _strip_extra_chars(prediction):
    return [value for value in prediction if value not in ('EOS', 'END', 'MASK')]


def plot_err_vs_seq_len(label_dict, err_dict, out_path):
    errors = []
    lengths = []
    for (id, label) in label_dict.items():
        err = err_dict[id]
        label_str = ''.join(_strip_extra_chars(label))
        label_len = len(label_str)

        errors.append(err)
        lengths.append(label_len)

    plt.plot(lengths, errors, 'gx', rasterized=False)

    plt.xlabel('Number of characters')
    plt.ylabel('Error rate [%]')

    plt.savefig(out_path, dpi=150, rasterized=False)
    plt.close()


def compute_uer_confusion_matrix(predictions_dict, labels_dict, unit_dict):

    slim_dict = {key:val for key, val in unit_dict.items() if val not in ['GO', 'EOS', 'MASK', 'END']}
    vocab_size = len(slim_dict)
    invdict = {v: k for k, v in slim_dict.items()}

    conf_matrix = np.zeros(shape=(vocab_size, vocab_size + 2))  # plus deletions, insertions
    edit_ops_indices = []
    edit_ops_at_word_boundaries = []
    edit_ops_not_at_word_boundaries = []

    for (id, label) in labels_dict.items():
        label_str = ''.join(_strip_extra_chars(label))
        prediction_str = ''.join(_strip_extra_chars(predictions_dict[id]))
        edit_ops = Levenshtein.editops(prediction_str, label_str)

        seen_positions = []
        for op in edit_ops:
            opname = op[0]
            if len(prediction_str) >= 40:
                edit_ops_indices.append(op[1] / len(prediction_str))  # store all errors in the source (prediction) string

            if opname == 'delete':
                source_unit = prediction_str[op[1]]
                mat_col = vocab_size
                seen_positions.append(op[1])

                if source_unit == ' ':
                    edit_ops_at_word_boundaries.append(source_unit)
                else:
                    edit_ops_not_at_word_boundaries.append(source_unit)

            elif opname == 'insert':
                source_unit = label_str[op[2]]  # the inserted unit does not exist in the source string
                mat_col = vocab_size + 1
            elif opname == 'replace':
                source_unit = prediction_str[op[1]]
                dest_unit = label_str[op[2]]
                mat_col = invdict[dest_unit] - 1
                seen_positions.append(op[1])

                if source_unit == ' ':
                    edit_ops_at_word_boundaries.append(source_unit)
                else:
                    edit_ops_not_at_word_boundaries.append(source_unit)

            else:
                raise Exception('unknown opname {}'.format(opname))

            mat_row = invdict[source_unit] - 1
            conf_matrix[mat_row, mat_col] += 1


        for idx, symbol in enumerate(prediction_str):
            if idx not in seen_positions:  # correct match
                mat_pos = invdict[symbol] - 1
                conf_matrix[mat_pos, mat_pos] += 1

    # plot_confusion_matrix(conf_matrix, invdict)
    plot_edit_ops_histogram(edit_ops_indices)


def plot_confusion_matrix(confusion_matrix, unit_dict):

    normed_matrix = confusion_matrix / confusion_matrix.sum(axis=0) * 100

    rows, cols = np.shape(confusion_matrix)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(normed_matrix, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        yticklabels=list(unit_dict.keys()),
        xticklabels=list(unit_dict.keys()) + ['Del', 'Ins'],
        yticks=np.arange(rows),
        xticks=np.arange(cols),
    )
    plt.show()

def plot_edit_ops_histogram(edit_ops_list):
    plt.hist(x=edit_ops_list, bins=100)
    # plt.plot(edit_ops_list, 'rx')
    plt.xlabel('normalised sentence length')
    plt.ylabel('number of errors')
    plt.savefig('errors.pdf')

    print('')
