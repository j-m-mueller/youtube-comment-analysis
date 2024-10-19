import sys

sys.path.insert(0, './src')

from comment_processor import CommentProcessor, plot_results


if __name__ == '__main__':
    with open('./html_body.txt', 'r', encoding='utf-8') as file:
        raw_html = file.read()
    
    cp = CommentProcessor()
    response_dict = cp.process_comments(raw_html=raw_html)
    
    plot_results(response_dict=response_dict,
                 plot_word_cloud=True)
