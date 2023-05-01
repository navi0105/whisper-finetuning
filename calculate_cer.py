import argparse
import evaluate
import pandas as pd
from utils.CER import CER
from ast import literal_eval

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--result-file",
        type=str,
        required=True,
        help=""
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=',',
        help=""
    )
    parser.add_argument(
        "--text-ref-column",
        type=str,
        default='lyric',
        help=""
    )
    parser.add_argument(
        "--text-pred-column",
        type=str,
        default='inference_lyric',
        help=""
    )
    parser.add_argument(
        "--phoneme-ref-column",
        type=str,
        default='phoneme_seq',
        help=""
    )
    parser.add_argument(
        "--phoneme-pred-column",
        type=str,
        default='inference_phoneme_seq',
        help=""
    )

    args = parser.parse_args()
    return args

def compute_cer(reference, prediction):
    CER_weighted = 0.0
    op_count = {'substitution': 0,
                'insertion': 0,
                'deletion': 0}
    for ref, pred in zip(reference, prediction):
        try:
            cer, nb_map = CER(hypothesis=list(pred),
                            reference=list(ref))

        except:
            cer, nb_map = CER(hypothesis=[],
                              reference=list(ref))
            
        CER_weighted += cer
        op_count['substitution'] += nb_map['S']
        op_count['insertion'] += nb_map['I']
        op_count['deletion'] += nb_map['D']
    
    print('=' * 30)
    print("CER (Weighted):", CER_weighted / len(reference))
    print("Wrong Operations:")
    for key, value in op_count.items():
        print(f"{key}: {value}")
    print('-' * 30)
    # weighted evaluate
    metric = evaluate.load("cer")
    CER_unweighted = metric.compute(references=reference,
                                  predictions=prediction)
    
    print("CER (Unweighted):", CER_unweighted)
    print("=" * 30)

def compute_per(reference, prediction):
    CER_weighted = 0.0
    op_count = {'substitution': 0,
                'insertion': 0,
                'deletion': 0}
    for ref, pred in zip(reference, prediction):
        try:
            cer, nb_map = CER(hypothesis=pred,
                             reference=ref)

        except:
            cer, nb_map = CER(hypothesis=[],
                              reference=ref)
            
        CER_weighted += cer
        op_count['substitution'] += nb_map['S']
        op_count['insertion'] += nb_map['I']
        op_count['deletion'] += nb_map['D']
    
    print('=' * 30)
    print("PER (Weighted):", CER_weighted / len(reference))
    print("Wrong Operations:")
    for key, value in op_count.items():
        print(f"{key}: {value}")
    print('-' * 30)
    # weighted evaluate
    # metric = evaluate.load("cer")
    CER_unweighted = (nb_map['S'] + nb_map['I'] + nb_map['D']) / (nb_map['S'] + nb_map['I'] + nb_map['C'])
    
    print("PER (Unweighted):", CER_unweighted)
    print("=" * 30)

def main():
    args = parse_args()
    
    result = pd.read_csv(args.result_file, sep=args.sep)

    compute_cer(result[args.text_ref_column], result[args.text_pred_column])
    compute_per(result[args.phoneme_ref_column].apply(lambda x: literal_eval(x)), 
                result[args.phoneme_pred_column].apply(lambda x: literal_eval(x)))



if __name__ == "__main__":
    main()