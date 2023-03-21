import argparse
import evaluate
import pandas as pd
from utils.CER import CER

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
        "--ref-column",
        type=str,
        default='lyric',
        help=""
    )
    parser.add_argument(
        "--pred-column",
        type=str,
        default='inference',
        help=""
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    result = pd.read_csv(args.result_file, sep=args.sep)
    reference = result[args.ref_column]
    prediction = result[args.pred_column]

    # non-weighted evaluate
    CER_unweighted = 0.0
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
            
        CER_unweighted += cer
        op_count['substitution'] += nb_map['S']
        op_count['insertion'] += nb_map['I']
        op_count['deletion'] += nb_map['D']
    
    print('=' * 30)
    print("CER (Unweighted):", CER_unweighted / len(reference))
    print("Wrong Operations:")
    for key, value in op_count.items():
        print(f"{key}: {value}")
    print('-' * 30)
    # weighted evaluate
    metric = evaluate.load("cer")
    CER_weighted = metric.compute(references=reference,
                                  predictions=prediction)
    
    print("CER (Weighted):", CER_weighted)
    print("=" * 30)





if __name__ == "__main__":
    main()