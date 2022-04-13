import os
from ThirdParty.ROUGE import pyrouge
from nltk import sent_tokenize
import shutil

def make_html_safe(s):
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s


def rouge(ref, hyp, log_path):
    assert len(ref) == len(hyp)
    ref_dir = os.path.join(log_path, 'reference')
    cand_dir = os.path.join(log_path, 'candidate')
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
    if not os.path.exists(cand_dir):
        os.makedirs(cand_dir)
    for i in range(len(ref)):
        with open(os.path.join(ref_dir, "%06d_reference.txt" % i), 'w', encoding='utf-8') as f:
            tokenized_ref = sent_tokenize(ref[i])
            tokenized_ref = '\n'.join(tokenized_ref)
            f.write(make_html_safe(tokenized_ref) + '\n')
        with open(os.path.join(cand_dir, "%06d_candidate.txt" % i), 'w', encoding='utf-8') as f:
            tokenized_cand = sent_tokenize(hyp[i])
            tokenized_cand = '\n'.join(tokenized_cand)
            f.write(make_html_safe(tokenized_cand) + '\n')

    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_candidate.txt'
    r.model_dir = ref_dir
    r.system_dir = cand_dir
    rouge_results = r.convert_and_evaluate()
    scores = r.output_to_dict(rouge_results)
    recall = [round(scores["rouge_1_recall"] * 100, 2),
              round(scores["rouge_2_recall"] * 100, 2),
              round(scores["rouge_l_recall"] * 100, 2)]
    precision = [round(scores["rouge_1_precision"] * 100, 2),
                 round(scores["rouge_2_precision"] * 100, 2),
                 round(scores["rouge_l_precision"] * 100, 2)]
    f_score = [round(scores["rouge_1_f_score"] * 100, 2),
               round(scores["rouge_2_f_score"] * 100, 2),
               round(scores["rouge_l_f_score"] * 100, 2)]
    print("F_measure: %s Recall: %s Precision: %s\n"
          % (str(f_score), str(recall), str(precision)))

    # remember to delete folder
    # with open(ref_dir + "rougeScore", 'w+', encoding='utf-8') as f:
    #     f.write("F_measure: %s Recall: %s Precision: %s\n"
    #             % (str(f_score), str(recall), str(precision)))

    print("deleting {}".format(ref_dir))
    shutil.rmtree(ref_dir)
    shutil.rmtree(cand_dir)

    return f_score[:], recall[:], precision[:]


def readline_aslist(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(line.strip().replace('\n',''))
    return data

import nltk
nltk.download('punkt')

if __name__ == '__main__':

    # dataset = "SummScreen"
    # hypo_file_name = 'test_reform_1024.hypo'
    # test_file_name = 'test.target'
    dataset = "../ICSI_label_stage4"
    hypo_file_name = 'test_adaptive.hypo'
    test_file_name = 'test.target'
    # dataset = "AMI"
    # hypo_file_name = 'test_label_stage2_0.7_beam4.hypo'
    # test_file_name = 'test_reform.target'
    ref = readline_aslist("./{}/{}".format(dataset, test_file_name))
    hypo = readline_aslist("./{}/{}".format(dataset, hypo_file_name))

    log_path = "./test_{}/".format(hypo_file_name)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    print("models: ", dataset, hypo_file_name)
    rouge(ref, hypo, log_path)

    shutil.rmtree(log_path)
