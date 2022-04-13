import nltk
from rouge import Rouge
from ThirdParty.rouge.rouge.rouge_score import *
from utils.tools import download_nltk

download_nltk()

class SourceSplitterCore(object):
    def __init__(self, token_per_seg):
        self.k = token_per_seg

    def segment_one_sample(self, sou):
        samples = []
        cur = []
        count = 0
        for i in range(len(sou)):
            trans = sou[i] # trans is a turn
            trans_len = len(trans.split(' '))
            trans = trans.replace('\n', ' ').replace('\r', '').replace('@@ ','')
            if count + trans_len > self.k:
                if count != 0: samples.append(cur)
                count = trans_len
                cur = [trans]
            else:
                cur.append(trans)
                count+=trans_len
        if len(cur):
            samples.append(cur)
        return samples


class TargetSplitterCore(object):
    def __init__(self, max_length: int = 100):
        self.max_length = max_length
        self.rouge = Rouge()

    def _get_rouge_from_ngram(self, reference_ngrams: Ngrams, evaluated_ngrams: Ngrams)-> dict:
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)

        # Gets the overlapping ngrams between evaluated and reference
        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)
        return f_r_p_rouge_n(evaluated_count, reference_count, overlapping_count)

    def _text_to_ngrams(self, text, n=1):
        ngrams = list(nltk.ngrams(nltk.word_tokenize(text), n))
        return Ngrams(ngrams)

    # This function is faster than seg_based_on_rouge because it uses the ngrams to computer rouge rather than text.
    def fast_rouge(self,sou, tar, name=None, verbose=False):
        cur_new = ''
        cur_ngram = Ngrams()
        best_score = 0
        best_sents = []

        # use ngram to represent each text
        sou = self._text_to_ngrams(sou)
        seg = [(x, self._text_to_ngrams(x), i) for i, x in enumerate(nltk.sent_tokenize(tar))]

        tot_len = len(seg)
        for i in range(min(self.max_length, tot_len)):
            scores = [(x, self._get_rouge_from_ngram(cur_ngram.union(seg_ngram), sou), i)
                      for x, seg_ngram, i in seg]
            best_seg = max(scores, key=lambda x: x[1]['f'])
            seg = [x for x in seg if x[2] != best_seg[2]]  # remove dup
            cur_new += ' ' + best_seg[0]
            cur_ngram = self._text_to_ngrams(cur_new)
            cur_score = self._get_rouge_from_ngram(cur_ngram, sou)['f']
            if cur_score > best_score:
                best_score = cur_score
                best_sents.append(best_seg)
            else:
                break

        if verbose:
            print("id:", name, "input/output:", tot_len, len(best_sents), "best:", best_score)
        best_string = list(set((x[0], x[2]) for x in best_sents))
        best_string.sort(key=lambda x: x[1])
        best_string = ' '.join([x[0] for x in best_string])

        return best_sents, best_string

    def seg_based_on_rouge(self, sou, tar, name=None, verbose=False) -> (list, str):
        cur_new = ''
        best_score = 0
        best_sents = []
        seg = [(x, i) for i, x in enumerate(nltk.sent_tokenize(tar))]
        tot_len = len(seg)
        for i in range(min(self.max_length, tot_len)):
            scores = [(x, self.rouge.get_scores(cur_new + ' ' + x, sou), i) for x, i in seg]
            scores.sort(key=lambda x: -x[1][0]['rouge-1']['f'])
            cur_new += scores[0][0] + ' '
            seg = [x for x in seg if x[1] != scores[0][2]]  # remove dup
            cur_score = self.rouge.get_scores(cur_new, sou)[0]['rouge-1']['f']
            if cur_score > best_score:
                best_score = cur_score
                best_sents.append(scores[0])
            else:
                break

        if verbose:
            print("id:", name, "input/output:", tot_len, len(best_sents), "best:", best_score)
        best_string = list(set((x[0], x[2]) for x in best_sents))
        best_string.sort(key=lambda x: x[1])
        best_string = ' '.join([x[0] for x in best_string])

        return best_sents, best_string

