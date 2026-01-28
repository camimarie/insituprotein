"""
Description: This file contains the Preprocessing class for downloading and processing proteome data from UniProt.
"""

import os
import re
import json
import requests

class Preprocessing:
    def __init__(self, data_directory="preprocessing/proteomes"):
        self.data_directory = data_directory
        os.makedirs(self.data_directory, exist_ok=True)

    def get_proteome(self, taxon_ID):
        """
        Downloads Swiss-Prot reviewed UniProt entries for a taxon of interest into a TSV.
        Entry<TAB>Sequence with header
        """
        tsv_path = os.path.join(self.data_directory, f'protein-list-{taxon_ID}.tsv')
        if os.path.exists(tsv_path):
            print(f"File {tsv_path} already exists.")
            return tsv_path

        session = requests.Session()
        retries = requests.adapters.Retry(
            total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504]
        )
        session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))

        query = f'(taxonomy_id:{taxon_ID}) AND (reviewed:true)'
        url = (
            'https://rest.uniprot.org/uniprotkb/search'
            f'?fields=accession,sequence&format=tsv&query={requests.utils.quote(query)}&size=500'
        )

        first_batch = True
        total_entries = 0

        while url:
            resp = session.get(url)
            resp.raise_for_status()
            lines = resp.text.splitlines()

            with open(tsv_path, 'a') as f:
                if first_batch:
                    print(lines[0], file=f)
                    lines = lines[1:]
                    first_batch = False
                else:
                    lines = lines[1:]

                for line in lines:
                    if not line.strip():
                        continue
                    print(line, file=f)

            total_entries += len(lines)
            print(f"Progress: {total_entries} / {resp.headers.get('x-total-results')}")

            # pagination
            link_header = resp.headers.get('Link', '')
            matches = re.findall(r'<([^>]+)>;\s*rel="next"', link_header)
            url = matches[0] if matches else None

        print(f"Completed TSV download: {tsv_path}")
        return tsv_path

    def tsv_to_json(self, taxon_ID):
        """
        Reads TSV and dumps to JSON in the format {"protein_id": {"sequence": "..."}}
        """
        tsv_path = os.path.join(self.data_directory, f'protein-list-{taxon_ID}.tsv')
        json_path = os.path.join(self.data_directory, f'protein-data-{taxon_ID}.json')

        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"TSV not found: {tsv_path}. Run get_proteome first.")

        data = {}
        with open(tsv_path, 'r') as fh:
            header = next(fh).strip().split('\t')
            if len(header) < 2:
                raise ValueError(f"Unexpected header: {header}")

            for line in fh:
                if not line.strip():
                    continue
                cols = line.strip().split('\t')
                if len(cols) < 2:
                    continue
                protein_id, sequence = cols[0], cols[1]
                data[protein_id] = {"sequence": sequence}

        with open(json_path, 'w') as f:
            json.dump(data, f)

        print(f"Wrote JSON with {len(data)} proteins: {json_path}")
        return json_path

    def check_entries(self, taxon_ID):
        """
        Verifies that TSV and JSON contain the same protein IDs.
        """
        json_path = os.path.join(self.data_directory, f'protein-data-{taxon_ID}.json')
        tsv_path = os.path.join(self.data_directory, f'protein-list-{taxon_ID}.tsv')

        if not (os.path.exists(json_path) and os.path.exists(tsv_path)):
            print("Required files missing.")
            return

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        with open(tsv_path, 'r') as f:
            next(f)
            tsv_ids = [line.split('\t')[0].strip() for line in f if line.strip()]

        missing_from_json = sorted(set(tsv_ids) - set(json_data.keys()))
        missing_from_tsv = sorted(set(json_data.keys()) - set(tsv_ids))

        if missing_from_json:
            print("Missing from JSON:", missing_from_json[:10])
        if missing_from_tsv:
            print("Missing from TSV:", missing_from_tsv[:10])

        if not missing_from_json and not missing_from_tsv:
            print("TSV and JSON entries match perfectly.")