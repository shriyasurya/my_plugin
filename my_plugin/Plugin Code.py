#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import spacy
from spacy.tokens import Doc
import pandas as pd
import networkx as nx
from fuzzywuzzy import fuzz


def merge_subjects(dataframe, threshold=80):
    # Calculate pairwise string similarity using Levenshtein distance
    similarity_matrix = [[fuzz.ratio(a, b) for b in dataframe['Subjects']] for a in dataframe['Subjects']]

    # Create a knowledge-based graph
    graph = nx.Graph()
    graph.add_nodes_from(dataframe['Subjects'])

    for i in range(len(dataframe)):
        for j in range(i + 1, len(dataframe)):
            if similarity_matrix[i][j] >= threshold:
                graph.add_edge(dataframe['Subjects'][i], dataframe['Subjects'][j])

    # Merge subjects based on connected components in the graph
    merged_subjects = []
    for component in nx.connected_components(graph):
        merged_subject = sorted(list(component), key=len)[0]  # Choose the shortest subject as the merged subject
        merged_subjects.append(merged_subject)

    # Update the dataframe with merged subjects
    merged_dataframe = dataframe.copy()
    merged_dataframe['Merged_Subject'] = merged_dataframe['Subjects'].apply(lambda x: get_merged_subject(x, merged_subjects))

    return merged_dataframe


def get_merged_subject(subject, merged_subjects):
    for merged_subject in merged_subjects:
        if any(subj in subject for subj in merged_subject.split(', ')):
            return merged_subject

    return subject


# Define a spaCy extension attribute for merged subjects
Doc.set_extension('merged_subjects', default=None)


# Define the spaCy pipeline component
def merge_subjects_component(doc):
    # Convert the DataFrame into spaCy Doc format
    dataframe = pd.DataFrame({'Subjects': doc._.merged_subjects})
    merged_dataframe = merge_subjects(dataframe, threshold=80)

    # Assign merged subjects to the spaCy Doc extension attribute
    doc._.merged_subjects = list(merged_dataframe['Merged_Subject'])

    return doc


class MergeSubjectsPlugin:
    def __init__(self, name='merge_subjects'):
        self.name = name

    def __call__(self, nlp):
        merge_subjects_component.name = self.name
        nlp.add_pipe(merge_subjects_component, name=self.name, last=True)
        return nlp

