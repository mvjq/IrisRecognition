
from utils.extractandenconding import extractFeature, matchingTemplate
from time import time
import argparse

# args
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str,
                    help="../CASIA1/")
parser.add_argument("--template_dir", type=str, default="./templates/CASIA1/",
                    help="./templates/CASIA1/")
parser.add_argument("--threshold", type=float, default=0.37,
                    help="Threshold for matching.")
args = parser.parse_args()

# timing
start = time()
print('\tStart verifying {}\n'.format(args.filename))
template, mask, filename = extractFeature(args.filename)
result = matchingTemplate(template, mask, args.template_dir, args.threshold)

# results 
if result == -1:
    print('\tNo registered sample.')
elif result == 0:
    print('\tNo sample found.')
else:
    print('\tsamples found (desc order of reliability):'.format(len(result)))
    for res in result:
        print("\t", res)
# total time
end = time()
print('\n\tTotal time: {} [s]\n'.format(end - start))