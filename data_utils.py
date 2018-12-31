from tensorflow.python.platform import gfile
import re
from nltk import word_tokenize
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
_DIGIT_RE = re.compile(r"\d")

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def create_vocabulary(vocab_path, vocab_size, file_path1, create_file=True):
    dict = {}
    summary = []
    sentences = []
    ct = 0

    with gfile.GFile(file_path1, mode="r") as f1:
        q = f1.readline().rstrip("\n")
        while q:
            q = f1.readline()
            q = f1.readline()
            # print(q)
            sentences = []
            while q != "\n":
                sen = word_tokenize(q.rstrip("\n"))
                sentences.append(sen)
                for word in sen:
                    w = _DIGIT_RE.sub(r"0", word.lower())
                    if w in dict:
                        dict[w] += 1
                    else:
                        dict[w] = 1
                q = f1.readline()
            ct += 1
            if ct % 100000 == 0:
                print("  reading data line %d" % ct)
            summary.append(sentences)
            q = f1.readline()
    # dict["..."] = 0
    vocab = _START_VOCAB + sorted(dict, key=dict.get, reverse=True)
    if len(vocab) > vocab_size:
        vocab = vocab[:vocab_size]
    if create_file:
        with gfile.GFile(vocab_path, mode="w") as vocab_file:
            for w in vocab:
                vocab_file.write(w + "\n")
    return vocab, summary




def prepare_data(data_dir, file1, vocab_size):
    story_set = []
    title_set = []
    vocab_path = data_dir + "/vocab_" + str(vocab_size)
    train_story_path = data_dir + "/train_story_" + str(vocab_size)
    train_title_path = data_dir + "/train_title_" + str(vocab_size)
    valid_story_path = data_dir + "/valid_story_" + str(vocab_size)
    valid_title_path = data_dir + "/valid_title_" + str(vocab_size)
    if not False:
        _, train_data = create_vocabulary(vocab_path, vocab_size, file1, True)
        vocab, _ = initialize_vocabulary(vocab_path)
        _, valid_data = create_vocabulary(vocab_path, vocab_size, "valid.txt", False)
        _, test_data = create_vocabulary(vocab_path, vocab_size, "test.txt", False)
        f = open("train.ids", "w", encoding="utf-8")
        for sentences in train_data:
            ids = []
            for s in sentences:
                ids.extend([str(vocab.get(_DIGIT_RE.sub(r"0", w.lower()), UNK_ID)) for w in s])
                ids.append(str(-1))
            f.write(" ".join(ids) + "\n")
        f.close()

        f = open("valid.ids", "w", encoding="utf-8")
        for sentences in valid_data:
            ids = []
            for s in sentences:
                ids.extend([str(vocab.get(_DIGIT_RE.sub(r"0", w.lower()), UNK_ID)) for w in s])
                ids.append(str(-1))
            f.write(" ".join(ids) + "\n")
        f.close()

        f = open("test.ids", "w", encoding="utf-8")
        for sentences in test_data:
            ids = []
            for s in sentences:
                ids.extend([str(vocab.get(_DIGIT_RE.sub(r"0", w.lower()), UNK_ID)) for w in s])
                ids.append(str(-1))
            f.write(" ".join(ids) + "\n")
        f.close()


    return train_title_path, train_story_path, valid_title_path, valid_story_path, vocab_path

