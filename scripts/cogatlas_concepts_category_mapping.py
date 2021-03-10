#%%
import json
import requests
from functools import reduce
from pathlib import Path

import pandas as pd
import tqdm
from bs4 import BeautifulSoup

#%%
def parse_mappings_from_categories_page():
    response = requests.get("https://www.cognitiveatlas.org/concepts/categories/all")

    content = BeautifulSoup(response.text, "html.parser")

    categories = content.find_all("h3")
    concepts_lists = content.find_all(class_="concepts-container")

    mapping_concept_to_category = {}
    mapping_category_to_concepts = {}
    for category, concepts_list in zip(categories, concepts_lists):
        category_name = category.text.split("Cassified under ")[1]
        concepts_names = [link.text for link in concepts_list.find_all("a")]

        for concept_name in concepts_names:
            mapping_concept_to_category[concept_name] = category_name
            mapping_category_to_concepts[category_name] = concept_name

    return mapping_concept_to_category, mapping_category_to_concepts

mapping_concept_to_category, mapping_category_to_concepts = parse_mappings_from_categories_page()

#%%
def get_cogatlas_concepts():
    url = "https://www.cognitiveatlas.org/api/v-alpha/concept?format=json"

    return requests.get(url).json()

cogatlas_concepts = get_cogatlas_concepts()

#%%
sample_labels_file = Path("../output") / "a7_label_all_syn_hyp.csv"
labels = pd.read_csv(sample_labels_file)

#%%
unique_labels = reduce(
    lambda acc, el: acc.union(el),
    list(labels.labels_all.dropna().str.split(", ").values),
    set()
)

#%%
missing_labels_in_mapping = [
    key for key in unique_labels
    if key not in mapping_concept_to_category.keys()
]

#%%
def find_cogatlas_concept_by_name(name, cogatlas_concepts):
    matches = [concept for concept in cogatlas_concepts if name == concept["name"]]

    if len(matches) == 0:
        print(f"No match found for {name}")
        return None

    if len(matches) > 1:
        print("Multiple matches. Returning the first one")

    return matches[0]


def fetch_cogatlas_parent_by_id(concept_id):
    """
    Fetch the "is a kind of" parent of the concept on CogAtlas website
    """
    url = f"https://www.cognitiveatlas.org/concept/id/{concept_id}"

    response = requests.get(url).text
    content = BeautifulSoup(response, "html.parser")

    potential_parent = content.find_all(class_="concept-assertion-list")
    parent = potential_parent[0].find("li")

    if parent and parent.find_all("a"):
        return parent.find_all("a")[1].text

    return None

analogy_id = find_cogatlas_concept_by_name("analogy", cogatlas_concepts)

#%%
potential_matched_concepts = {}
for missing_concept in tqdm.tqdm(missing_labels_in_mapping):
    missing_concept_id = find_cogatlas_concept_by_name(missing_concept, cogatlas_concepts)

    if missing_concept_id is not None:
        potential_matched_concepts[missing_concept] = fetch_cogatlas_parent_by_id(missing_concept_id["id"])

#%%
for _ in range(3):
    for concept_name, matched_concept in potential_matched_concepts.items():
        if (concept_name not in mapping_concept_to_category) and (matched_concept in mapping_concept_to_category):
            mapping_concept_to_category[concept_name] = mapping_concept_to_category[matched_concept]


#%%
still_missing_labels = [label for label in missing_labels_in_mapping if label not in mapping_concept_to_category]

#%%
with open("cogatlas_concepts_categories_mapping.json", "w+") as file:
    json.dump(mapping_concept_to_category, file)
