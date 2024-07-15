import argparse

parser = argparse.ArgumentParser(description='optimizing model parameters')
parser.add_argument('--type', choices=['random', 'genetic', 'anneal'],
                    default='regression', help='gives the type of optimizer, can be random, genetic, or anneal (simulated annealing)')
args = parser.parse_args()

if args.type == 'random':
    for i in range(100):
        print('wow')