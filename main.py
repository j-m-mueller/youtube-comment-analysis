import argparse
import sys

sys.path.insert(0, './src')

from comment_processor import CommentProcessor, plot_results


def main():
    # parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw-html-path', 
                        type=str, 
                        default='./html_body.txt',
                        help='Path to the raw HTML document (optional)')
    parser.add_argument('--plot-results', 
                        choices=['true', 'false'],
                        default='true',
                        help='Trigger plotting of the analysis results')
    parser.add_argument('--plot-word-cloud',
                        choices=['true', 'false'],
                        default='true',
                        help='Generate a word cloud for positive and negative comments')

    args = parser.parse_args()
    print(f"\nExecuting analysis with arguments: {args}\n")

    # load data
    with open(args.raw_html_path, 'r', encoding='utf-8') as file:
        raw_html = file.read()

    # process data
    cp = CommentProcessor()
    response_dict = cp.process_comments(raw_html=raw_html)

    # plot results
    if args.plot_results == 'true':
        plot_results(response_dict=response_dict,
                     plot_word_cloud=args.plot_word_cloud == 'true')


if __name__ == '__main__':
    main()
