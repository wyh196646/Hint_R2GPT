import argparse
import math
parser = argparse.ArgumentParser()

## nargs='+'接受1个或多个参数，
## nargs='*'接受零个或多个
parser.add_argument('--list',type=int, nargs='+', help='<Required> Set flag', required=True)
args = parser.parse_args()

if __name__ == '__main__':
    print(args.list)
    print(sum(args.list))

