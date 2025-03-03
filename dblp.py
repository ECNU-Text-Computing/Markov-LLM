import ijson
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import json
from decimal import Decimal

def save_to_json(data_dict, file_name):

    def set_default(obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, Decimal):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(data_dict, file, ensure_ascii=False, indent=4, default=set_default)
    print(f"Data successfully saved to {file_name}")

if __name__ == '__main__':
    ref_dict = defaultdict(dict)

    with open('class/dblp_v14.json', 'rb') as f:
        rows = ijson.items(f, 'item')
        for paper in tqdm(rows):
            if 'id' in paper and 'year' in paper:
                ref_dict[paper['id']] = {
                    'paper': paper,
                    'references': set(paper.get('references', [])),
                    'cited_by': set(),
                    'cited_by_10yr': set(),
                    'ref_cited_by': set(),
                    'ref_cited_by_10yr': set()
                }
        print('读取论文完毕')
    save_to_json(ref_dict, 'ref_dict_initial.json')

    # First loop for cited_by and ref_cited_by_10yr calculation
    for re_id, re_data in tqdm(list(ref_dict.items())):
        paper_year = re_data['paper'].get('year')
        if paper_year is not None:
            for ref_id in re_data['references']:
                ref_data = ref_dict.get(ref_id)
                if ref_data:
                    ref_paper_year = ref_data['paper'].get('year')
                    if ref_paper_year:
                        if paper_year - 5 <= ref_paper_year <= paper_year:
                            ref_data['cited_by'].add(re_id)  # For cited within 5 years
                        if paper_year - 10 <= ref_paper_year <= paper_year:
                            ref_data['cited_by_10yr'].add(re_id)  # For cited within 10 years
    print('预处理论文cit完毕')
    save_to_json(ref_dict, 'ref_dict_after_first_loop.json')

    # Second loop for refining cited_by collections based on year constraints
    for re_id, re_data in tqdm(ref_dict.items()):
        paper_year = re_data['paper'].get('year')
        if paper_year is not None:
            for ref_id in re_data['references']:
                if ref_id in ref_dict:
                    common_cited_by = ref_dict[ref_id]['cited_by'].copy()
                    common_cited_by_10yr = ref_dict[ref_id]['cited_by_10yr'].copy()
                    for id_x in common_cited_by.copy():
                        if id_x in ref_dict:
                            if paper_year <= ref_dict[id_x]['paper'].get('year', 0) <= paper_year + 5:
                                continue
                            else:
                                common_cited_by.discard(id_x)
                        if id_x in ref_dict and paper_year <= ref_dict[id_x]['paper'].get('year', 0) <= paper_year + 10:
                            continue
                        else:
                            common_cited_by_10yr.discard(id_x)
                    re_data['ref_cited_by'].update(common_cited_by)
                    re_data['ref_cited_by_10yr'].update(common_cited_by_10yr)
    print('预处理ref的cit完毕')
    save_to_json(ref_dict, 'ref_dict_after_second_loop.json')

    # Gathering data for DataFrame
    results = []
    for re_id, re_data in tqdm(ref_dict.items()):
        paper_id = re_data['paper'].get('id')
        if not paper_id:
            continue
        title = re_data['paper'].get('title', '')
        paper_ref = re_data['references']
        cited_by_papers = re_data['cited_by']
        cited_by_papers_10 = re_data['cited_by_10yr']
        ref_cited_by_papers = re_data['ref_cited_by']
        ref_cited_by_10yr_papers = re_data['ref_cited_by_10yr']
        abstract = re_data['paper'].get('abstract', '')
        year = re_data['paper'].get('year')
        if not abstract or not year:
            continue
        author_count = len(re_data['paper'].get('authors', []))

        if len(paper_ref) == 0 and len(cited_by_papers) == 0:
            continue

        # Calculate CD for 1-5 and 1-10 years
        s = len(cited_by_papers & ref_cited_by_papers)
        ni = len(cited_by_papers) - s
        nj = s
        nk = len(ref_cited_by_papers) - s
        d = (ni - nj) / (ni + nj + nk) if (ni + nj + nk) != 0 else 3

        s_new = len(cited_by_papers_10 & ref_cited_by_10yr_papers)
        ni_new = len(cited_by_papers_10) - s_new
        nj_new = s_new
        nk_new = len(ref_cited_by_10yr_papers) - s_new
        d_new = (ni_new - nj_new) / (ni_new + nj_new + nk_new) if (ni_new + nj_new + nk_new) != 0 else 3

        citation = len(cited_by_papers)
        ref_titles = [ref_dict[ref_id]['paper'].get('title', '') for ref_id in paper_ref if ref_id in ref_dict]
        ref_ids = list(paper_ref)
        ref_abs = [ref_dict[ref_id]['paper'].get('abstract', '') for ref_id in paper_ref if ref_id in ref_dict]

        results.append({
            'paper_id': paper_id, 'ni': ni, 'nj': nj, 'nk': nk, 'd': d, 'd_new': d_new,
            'citation': citation, 'title': title, 'abstract': abstract, 'year': year,
            'author_count': author_count, 'ref_titles': ref_titles, 'ref_ids': ref_ids, 'ref_abs': ref_abs
        })

    df = pd.DataFrame(results)
    df = df.loc[df['abstract'].str.count('\s+') > 20]  # Filter papers with short abstracts
    df.to_csv('dblp_review.csv', index=False, escapechar='\\')
    print('Data processing and saving complete')