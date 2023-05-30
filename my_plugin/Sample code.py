#!/usr/bin/env python
# coding: utf-8

# ### Imports

# #### This is a sample notebook which takes sentences from tweets converts them into beliefs and the subjects are extracted from those beliefs

# In[19]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
# import pretrainer
import pickle
from sentence_transformers import SentenceTransformer
import re
import spacy
from spacy_langdetect import LanguageDetector
from spacy.language import Language
#from allTokens import *
from langdetect import detect
from dataclasses import dataclass
from pydoc import resolve
from tabnanny import verbose
#from tkinter import BEVEL
import spacy
from spacy.tokens import Span, Token, Doc
from spacy.matcher import Matcher, DependencyMatcher
from spacy.util import filter_spans
import typing

#from negspacy.negation import Negex
from spacy.language import Language
from negspacy.termsets import termset
import re
import coreferee
#from my_negex import MyNegex
from dataclasses import dataclass
from pydoc import resolve
from tabnanny import verbose
#from tkinter import BEVEL
import spacy
from spacy.tokens import Span, Token, Doc
from spacy.matcher import Matcher, DependencyMatcher
from spacy.util import filter_spans
import typing

from negspacy.negation import Negex
from spacy.language import Language
from negspacy.termsets import termset
import networkx as nx
from fuzzywuzzy import fuzz


# ### Preprocessing

# In[3]:


emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
abbr_dict={
    "what's":"what is",
    "what're":"what are",
    "who's":"who is",
    "who're":"who are",
    "where's":"where is",
    "where're":"where are",
    "when's":"when is",
    "when're":"when are",
    "how's":"how is",
    "how're":"how are",

    "i'm":"i am",
    "we're":"we are",
    "you're":"you are",
    "they're":"they are",
    "it's":"it is",
    "he's":"he is",
    "she's":"she is",
    "that's":"that is",
    "there's":"there is",
    "there're":"there are",

    "i've":"i have",
    "we've":"we have",
    "you've":"you have",
    "they've":"they have",
    "who've":"who have",
    "would've":"would have",
    "not've":"not have",

    "i'll":"i will",
    "we'll":"we will",
    "you'll":"you will",
    "he'll":"he will",
    "she'll":"she will",
    "it'll":"it will",
    "they'll":"they will",
    "isn't":"is not",
    "wasn't":"was not",
    "aren't":"are not",
    "weren't":"were not",
    "can't":"can not",
    "couldn't":"could not",
    "don't":"do not",
    "didn't":"did not",
    "shouldn't":"should not",
    "wouldn't":"would not",
    "doesn't":"does not",
    "haven't":"have not",
    "hasn't":"has not",
    "hadn't":"had not",
    "won't":"will not",
    "you'd": "you would",
    "he's": "he is",
    "I'd":"I would"
}


# In[4]:


my_nlp = None
extraChar = {'&quot;': '"',
 '&amp;': 'and',
 '&lt;': '<',
 '&gt;': '>',
 '&nbsp;': 'un-linebreak-able space',
 '&iexcl;': '¡',
 '&cent;': '¢',
 '&pound;': '£',
 '&curren;': '¤',
 '&yen;': '¥',
 '&brvbar;': '¦',
 '&sect;': '§',
 '&uml;': '¨',
 '&copy;': '©',
 '&ordf;': 'ª',
 '&laquo;': '«',
 '&not;': '¬',
 '&shy;': '\xad',
 '&reg;': '®',
 '&macr;': '¯',
 '&deg;': '°',
 '&plusmn;': '±',
 '&sup2': '²',
 '&sup3;': '³',
 '&acute;': '´',
 '&micro;': 'µ',
 '&para;': '¶',
 '&middot;': '·',
 '&cedil;': '¸',
 '&sup1;': '¹',
 '&ordm;': 'º',
 '&raquo;': '»',
 '&frac14;': '¼',
 '&frac12;': '½',
 '&frac34;': '¾',
 '&iquest;': '¿',
 '&times;': '×',
 '&divide;': '÷',
 '&ETH;': 'Ð',
 '&eth;': 'ð',
 '&THORN;': 'Þ',
 '&thorn;': 'þ',
 '&AElig;': 'Æ',
 '&aelig;': 'æ',
 '&OElig;': 'Œ',
 '&oelig;': 'œ',
 '&Aring;': 'Å',
 '&Oslash;': 'Ø',
 '&Ccedil;': 'Ç',
 '&ccedil;': 'ç',
 '&szlig;': 'ß',
 '&Ntilde;': 'Ñ',
 '&ntilde;': 'ñ'}

special = {
    "’":"'",
    "‘":"'",
    "`":"'",
    '“':'"',
    '”':'"',
    '…':"."
}
@Language.factory('language_detector')
def language_detector(nlp, name):
    return LanguageDetector()

def get_nlp():
    global my_nlp
    if not my_nlp:        
        my_nlp = spacy.load('en_core_web_lg')
        my_nlp.add_pipe('language_detector')
    return my_nlp

def is_english(text):
    doc = get_nlp()(text)
    return doc._.language['score']>0.95
        
def removeTags(text,splitter):
    div = text.split(splitter)
    endExists = True
    i = len(div)-1
    while i>=0 and endExists:
        if len(div[i].strip().split(" "))  == 1:
            div.pop(i)
            i-=1
        else:
            endExists = False
        
    return " "+splitter.join(div).strip()

def removeTagsFromStart(text,splitter):
    div = text.split(splitter)
    endExists = True
    i = 0
    while len(div)>0 and endExists:
        if len(div[i].strip().split(" "))  == 1:
            div=div[i+1:]
        else:
            div[i] = splitter+div[i]
            endExists = False
    if len(div) == 0:
        return ''
    splitfirst = div[0].split(" ")
    if "you" in splitfirst[1].lower() and "@" in splitfirst[0]:
        splitfirst[1] = splitfirst[0]
        splitfirst = splitfirst[1:]
        div[0] = " ".join(splitfirst)
    if "@" in splitfirst[0].strip()[0:2]:
        splitfirst = splitfirst[1:]
        div[0] = " ".join(splitfirst)
    return " ".join(div).strip()


def removeRT(text):
    if text[0:2] == 'RT':
        return ":".join(text.split(":")[1:]).strip()
    return text  

def clean_tweet(text,removeFromMiddle):
    text = text.strip()
    for key,value in special.items():
        text = re.sub(key,value,text)
    for key,value in abbr_dict.items():
        text = re.sub(key,value,text,flags=re.I)
    for key,value in extraChar.items():
        text = re.sub(key,value,text)
    
        #print(text)
    if removeFromMiddle:
        text = re.sub("@[A-Za-z0-9_]+","", text)
        text = re.sub("#[A-Za-z0-9_]+","", text)
    text = re.sub(r"http\S+", "", text)
    text = emoji_pattern.sub(r' ', text)
    text = removeTags(text,"#")
    text = removeTags(text,"@")
    text = removeTags(text,"#")
    text = removeTagsFromStart(text,"@")
    text = removeRT(text)
    text = re.sub(' +', ' ', text)
    text = re.sub("@",'',text)
    text = re.sub("#",'',text)
    text = re.sub(r'[\n\r]+',r'\n',text)
    text = re.sub('(?<![.?!])\n',". ",text)
    text = re.sub('\n'," ",text)
    #text = ' '.join(text.replace('\r', ' ').split())
    text = re.sub("\s+"," ",text)
    #text = re.sub(r"[^A-Za-z.!?'', ]",'',text)
    
    if not is_english(text):
        return ''
    return text.strip()


# In[5]:


#%run "preprocessing.py"
data = pd.read_csv("cleanedData1.csv")
d = data[['tweet_hash','full_tweet_text']]
d['cleaned'] = data['full_tweet_text'].apply(clean_tweet,args=(False,))


# In[6]:


d.head(5)


# In[7]:


default_ts = termset("en_clinical").get_patterns()
@Language.factory(
        "my_negex",
        default_config={
        "neg_termset": default_ts,
        "ent_types": list(),
        "extension_name": "my_negex",
        "chunk_prefix": list(),
    },
)
class MyNegex:
        def __init__(
                self,
                nlp:Language,
                name: str,
                neg_termset: dict,
                ent_types: list,
                extension_name: str,
                chunk_prefix: list,
        ):
                self.name = name
                self.extension_name = extension_name
                self.wrapped = Negex(nlp,"negex",neg_termset,ent_types,extension_name,chunk_prefix)
        
        def my_negex(self,doc):
                n = self.wrapped
                preceding, following, terminating = n.process_negations(doc)
                boundaries = n.termination_boundaries(doc, terminating)
                for b in boundaries:
                        sub_preceding = [i for i in preceding if b[0] <= i[1] < b[1]]
                        sub_following = [i for i in following if b[0] <= i[1] < b[1]]
                
                # THIS IS THE SINGLE CHANGE!!!!
                        for nc in doc[b[0] : b[1]].noun_chunks:
                                
                                if any(pre < nc.start for pre in [i[1] for i in sub_preceding]):
                                        nc._.set(self.extension_name, True)
                                        continue
                                if any(fol > nc.end for fol in [i[2] for i in sub_following]):
                                        nc._.set(self.extension_name, True)
                                        continue
                                if n.chunk_prefix:
                                        if any(
                                                nc.text.lower().startswith(c.text.lower())
                                                for c in n.chunk_prefix
                                        ):
                                                nc._.set(self.extension_name, True)
                return doc
        
        
        def __call__(self,doc):
                return self.my_negex(doc)


# In[8]:


Doc.set_extension("beliefs", default=[], force=True)
Span.set_extension("beliefs", default=[], force=True)

mental_state_verbs = "believe, think, doubt, fear, feel, think, know, understand, imagine, realize, recognize, mean, remember, forget, suppose, want, intend, try, aim, need, prefer".split(", ")
mental_state_verbs += "like, love, hate, dislike, want, hope, wish, feel, need, prefer, enjoy, appreciate, fear, envy, care, mind".split(", ")
mental_state_verbs += "agree, disagree".split(", ")
mental_state_verbs = set(mental_state_verbs)

class VerbChunk:
    def __init__(
        self,
        verb_span: Span,
        # Object side refers to the token in the span that should be used as the 
        # parent of the subject
        subject_side: Token,
        # Object side refers to the token in the span that should be used as the 
        # parent of the object
        object_side: Token
    ):
        self.verb_span = verb_span
        self.object_side = object_side
        self._subject_side = subject_side
        self.attributional = is_mental_state_verb(self.subject_side)
    
    @property
    def subject_side(self):
        return self._subject_side

    @subject_side.setter
    def subject_side(self,new_verb):
        self._subject_side = new_verb
        self.attributional = is_mental_state_verb(self._subject_side)

    def __repr__(self):
        return f"{self.verb_span} - Subject Root: {self.subject_side} Object Root: {self.object_side} Attributional: {self.attributional}"
    
    def to_dict(self):
        return {
            "verb_span":self.verb_span,
            "object_side":self.object_side,
            "subject_side":self._subject_side,
            "attributional":self.attributional
        }

@dataclass
class ProxyToken:
    text: str

class Belief:
    def __init__(
        self,
        subject: Span,
        verb: VerbChunk,
        object: Span,

    ):
        self.resolved_subject = None
        self.subject = subject
        self.verb = verb
        self.object = object
        self.nested_beliefs = []
        self.negated = False
        

    @property
    def subject(self):
        return self._subject
    
    @subject.setter
    def subject(self,nsubj):
        self._subject = nsubj
        self.resolved_subject = self.resolve_subject(nsubj)
       
    
    def resolve_subject(self,subj):
        result = []
        for ot in subj:

            if ot.pos_ in ("PRON"):
                toks = ot.doc._.coref_chains.resolve(ot)
                if toks:
                    result = [] 
                    for i,t in enumerate(toks):
                        if i>0:
                            result.append(ProxyToken("and"))
                        if ot.morph.to_dict().get("Poss")=="Yes":
                            result.append(ProxyToken(t.text+"'s"))
                        else:
                            result.append(t)
                elif ot.lemma_ == "who" and ot.doc[max(0,ot.i-1)].pos_=="PROPN":
                    result.append(ot.doc[ot.i-1])
                else: 
                    # We found a pronoun that can't be resolved, so bail
                    return None
            else:
                result.append(ot)
        return result
        

   
    def handle_negation(self):
        result = [t for t in self.verb.verb_span]
        if self.negated:
            #print(f"Want to negate {self.verb.verb_span.root} at {self.verb.verb_span.root.i}")
           
            auxpos = _find_pos(self.verb.verb_span,"AUX")
            if auxpos > -1:
                result.insert(auxpos+1,ProxyToken("not"))
            elif self.verb.verb_span.root.pos_ == 'VERB' and self.verb.verb_span.root.morph.to_dict().get("Tense") == "Past":
                pos = self.verb.verb_span.root.i - self.verb.verb_span.start
                del result[pos]
                result[pos:pos] = [ProxyToken("did"),ProxyToken("not"),ProxyToken(self.verb.verb_span.root.lemma_)]
                #print(f"Verb {v.text} lemma {v.lemma_} morph {v.morph}")
            else:
                result.insert(0,ProxyToken("not"))

        return result

    def smart_join(self,l,end = False):
        result = []
        
        for i,s in enumerate(l):
            if len(s.text.strip())==0:
                continue
            elif i==0 or (type(s)==Token and ((s.pos_=="PART" and not re.search('[aeiou]',s.text) or (s.pos_=="PUNCT" and s.text not in ["'",'"'])))):
                result.append(s.text)
            else:
                result.append(" ")
                result.append(s.text)
        if end and result[-1] not in ["!",".","?",'"',"'",")"]:
            result.append(".")
        return "".join(result)



    def as_string(self):
       
        tokens = [t for t in (self.resolved_subject if self.resolved_subject else self.subject)]
        tokens += [t for t in self.handle_negation()]
        tokens += [t for t in self.object] if self.object else []
        return self.smart_join(tokens).capitalize()


    def _generate_string(self,indent = ""):
        s =  f"{indent}Subj:{self.subject} : Verb:{self.verb.verb_span} : Object:{self.object}"
        if self.is_attributional():
            s = f"{s} [ATTRIBUTED]"
        if self.negated:
            s = f"{s} [NEGATED]"
        if self.nested_beliefs:
            for b in self.nested_beliefs:
                s = f"{s}-->\n{b._generate_string(f'{indent}    ')}"
        return s

    def __repr__(self):
        return self._generate_string()

    def mark_negated(self,b):
        self.negated = b

    def is_attributional(self):
        return self.verb.attributional

    def is_self_belief(self):
        return not self.nested_beliefs

    def clean_subject(self):
        tokens = [tok for tok in self.subject]
        result = []

        #TODO: Refactor this with resolve_subject
        for ot in tokens:

            if ot.pos_ in ("PRON"):
                toks = ot.doc._.coref_chains.resolve(ot)
                if toks:
                    result = [] 
                    for i,t in enumerate(toks):
                        if i>0:
                            result.append(ProxyToken("and"))
                        if ot.morph.to_dict().get("Poss")=="Yes":
                            result.append(ProxyToken(t.text+"'s"))
                        else:
                            result.append(t)
                elif ot.lemma_ == "who" and ot.doc[max(0,ot.i-1)].pos_=="PROPN":
                    result.append(ot.doc[ot.i-1])
                else: 
                    result.append(ot)
            else:
                result.append(ot)
        
        return " ".join([t.text for t in result])


    def clean_belief(self):
        result = []
        #print(f"Processing '{self.subject}' is attributional {self.is_attributional()}")
        
        # take care of deictic references
        det = _find_matching_child(self.subject.root,["det"])
       
        if det and det.root.morph.to_dict().get("PronType") == "Dem" and det.root.morph.to_dict().get("Number")=="Sing":
            #print("Excluding because of deixis")
            result = []
        
        elif self.is_attributional():
            #print(f"Processing attributional layer with person {self.subject.root.morph.to_dict().get('Person')}")
            if self.subject.root.morph.to_dict().get("Person") == '1':
                result = [sb.clean_belief() for sb in self.nested_beliefs]
            else:
                result = [(self.as_string(),self.clean_subject())]    
         
        elif self.subject.root.lemma_ in ["it","there"]:
            #print("Excluding because of it / there")
            result = [sb.clean_belief() for sb in self.nested_beliefs]
        
        # Remove pronouns
        elif self.subject.root.pos_ == "PRON" and not self.resolved_subject and not self.subject.root.lemma_=="we":            
            result = []
        
        else:
            result += [(self.as_string(),self.clean_subject())]
        
        return flatten(result)
    
    def to_dict(self):
        return {
            "resolved_subject":self.resolved_subject,
            "subject":self.subject,
            "verb":self.verb.to_dict(),
            "nested_beliefs":[x.to_dict() for x in self.nested_beliefs],
            "negated":self.negated
        }


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def _find_pos(span,pos):
    if not isinstance(pos,list):
        pos = [pos]
    for i,x in enumerate(span):
        if x.pos_ in pos:
            return i
    return -1
       

def _get_verb_matches(span):
    # 1. Find verb phrases in the span
    # (see mdmjsh answer here: https://stackoverflow.com/questions/47856247/extract-verb-phrases-using-spacy)
    match_keys = {
        1:"deep_match",
        2:"aux_verb_phrase",
        3:"idiomatic_wants",
        4:"plain_verb",
        5:"conj_verb",
        6:"xing_to_y"}

    deep_matcher = DependencyMatcher(span.vocab)
    pattern = [
    {
        "RIGHT_ID": "active_verb",
        "RIGHT_ATTRS": {"POS": {"IN":["VERB","AUX"]}}
    },
    {
        "LEFT_ID": "active_verb",
        "REL_OP": ">",
        "RIGHT_ID": "operating_aux",
        "RIGHT_ATTRS": {"POS": "AUX","DEP":{"NOT_IN":["ccomp","conj"]}}
    }]
    deep_matcher.add(1,[pattern])

    pattern = [
    {
        "RIGHT_ID": "active_verb",
        "RIGHT_ATTRS": {"POS": {"IN":["VERB","AUX"]}}
    },
    {
        "LEFT_ID": "active_verb",
        "REL_OP": ">",
        "RIGHT_ID": "comp_verb",
        "RIGHT_ATTRS": {"POS": "VERB","DEP":"xcomp"}
    },
    {
        "LEFT_ID": "comp_verb",
        "REL_OP": ">",
        "RIGHT_ID": "part",
        "RIGHT_ATTRS": {"POS": "PART"}
    }]
    deep_matcher.add(6,[pattern])

    # LL: Not sure if this helps or conflicts with the previous pattern
    # pattern = [
    #     {
    #         "RIGHT_ID": "subject_action",
    #         "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "VERB", 'AUX']}}
    #     },
    #     {
    #         "LEFT_ID": "subject_action",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "is",
    #         "RIGHT_ATTRS": {"POS": {"IN": ["AUX", "VERB"]}, "DEP": "advcl"}
    #     },
    #     {
    #         "LEFT_ID": "subject_action",
    #         "REL_OP": ">",
    #         "RIGHT_ID": "suject",
    #         "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", 'ADJ']}, "DEP": "amod"}
    #     },
    # ]
    #
    # deep_matcher.add("long_subject_action", [pattern])

    matches = deep_matcher(span)
    matched_spans = [(match_keys[m[0]],span[min(m[1]):max(m[1])+1]) for m in matches]
    verb_matcher = Matcher(span.vocab)
    ## JEI: Adding more complex verbs
    verb_matcher.add(2, [
        [{"POS": "AUX", "OP":'+'}, {"POS": "VERB", "OP":"?"}]])

    # JEI: Matching constructions like <Person> would have <Person> <verb>
    verb_matcher.add(3, [[
        {"ORTH": "would"},
        {"ORTH":"have"},
        {"POS": {"IN":["PROPN","PRON"]}},
        {"POS": {"IN":["VERB","AUX"]}}]])

    verb_matcher.add(4, [[{"POS": "VERB"}]])

    verb_matcher.add(5, [[{"POS": {"IN":["VERB","AUX"]}},{"ORTH":"and"},{"POS": {"IN":["VERB","AUX"]}}]])

    matches = verb_matcher(span)
    matched_spans += [(match_keys[id],span[start:end]) for id, start, end in matches]

    return matched_spans

def _get_maximal_spans(spans):
    spans = sorted(spans,key=lambda x:x.start)
    merged = []
    for s in spans:
        if not merged:
            merged.append(s)
        elif merged[-1].end < s.start:
            merged.append(s)
        elif s.end > merged[-1].end:
            merged[-1] = Span(s.doc,merged[-1].start,s.end)
    return merged

def _get_verb_chunks(span) -> typing.List[VerbChunk]:
    matches = _get_verb_matches(span)
  
    # Filter matches (e.g. do not have both "has won" and "won" in verbs)
    verb_chunks = []

    # Screening out adverbial clauses
    filtered_spans = [x for x in _get_maximal_spans([m[1] for m in matches]) if x.root.dep_!="advcl"]
    
   


    for f in filtered_spans:
        last = None
        first = None
        for t in f:
            if t.pos_ in ["VERB","AUX"]:
                if first is None:
                    first = t
                last = t
        # For now, marking subject & object side based on start and end of the span
        # Might need to be smarter - this is one of those places where understanding
        # how the dependency tree is built would help!
        vs = Span(f[0].doc,f[0].i,f[-1].i+1)
        ss,os = first,last
        if ss.dep_ == "auxpass":
            ss = ss.head
        
        if vs.root.dep_ == "ccomp" and vs.root.head.i < vs.root.i:
           os = vs.root.head
        
        verb_chunks.append(VerbChunk(vs,ss,os))
    return verb_chunks
        

def _update_subject_side(verb_chunk):
    root = verb_chunk.subject_side
    
    while root:
        # Can we find subject at current level?
        for c in root.children:
            if c.dep_ in ["nsubj", "nsubjpass","expl"]:
                subject = extract_span_from_entity(c)
                verb_chunk.subject_side = root
                return subject

        # ... otherwise recurse up one level
        if (root.dep_ in ["conj", "cc", "advcl", "acl", "ccomp", "aux"]
            and root != root.head):
            root = root.head
        else:
            root = None

    return None

def is_mental_state_verb(verb: Token):
    return verb.lemma_ in mental_state_verbs

def _find_matching_child(root, allowed_types):
    for c in root.children:
        if c.dep_ in allowed_types:
            return extract_span_from_entity(c)
    return None

def spans_intersect(span_a,span_b):
    return (span_a.start >= span_b.start and span_a.start < span_b.end) or         (span_b.start >= span_a.start and span_b.start < span_a.end)


def extract_span_from_entity(token):
    ent_subtree = sorted([c for c in token.subtree], key=lambda x: x.i)
    return Span(token.doc, start=ent_subtree[0].i, end=ent_subtree[-1].i + 1)

def generate_belief_for_chunk(vc:VerbChunk) -> Belief:
  
    subject = _update_subject_side(vc)
    if not subject:
        return None
    object_tokens = set()
    for v in vc.object_side.rights:
        object_tokens.update([t for t in v.subtree])
        #if v.i >= vc.verb_span.end:
            
    
    tl = sorted(object_tokens,key=lambda x:x.i)
    result = Belief(subject,vc,Span(subject.doc,max(tl[0].i,vc.verb_span.end),tl[-1].i+1) if tl else None)
    return result
    

def extract_beliefs(span):
    #print(f"Processing {span}")
    verb_chunks = _get_verb_chunks(span)
   
    beliefs = [b for b in [generate_belief_for_chunk(vc) for vc in verb_chunks] if b is not None]
    beliefs = _parse_conjunctions(beliefs) 
    beliefmap = {}
    
    #TODO check if verb is nested in object - if so, remove
    for b in beliefs:
        if b.object:
            beliefmap[(b.subject.start,b.object.start,b.object.end)] = b
        else:
            beliefmap[(b.subject.start,b.verb.verb_span.start,b.verb.verb_span.end)] = b
    #print(beliefmap)
    noparent_children = [p for p in beliefmap.keys()]
  
    for child in list(noparent_children):
        bestparent = None
        for parent in beliefmap.keys():
            if parent == child:
                continue

            # Going to find the shortest parent that contains this child
            if child[2] <= parent[2] and (child[0] >= parent[1] or child[1] > parent[1]):
                if child in noparent_children:
                    noparent_children.remove(child)
                if bestparent is None or bestparent[2]-bestparent[1] > parent[2] - parent[1]:
                    bestparent = parent
        if bestparent:
            beliefmap[bestparent].nested_beliefs.append(beliefmap[child])
    return [beliefmap[c] for c in noparent_children]

def _spans_cconj(token):
    #print(f"Token span should be {token.head}:{token.head.i} to {token}:{token.i}")
    for i in range(token.head.i,token.i):
        if token.doc[i].dep_ == "CCONJ":
            return True
    return False


def _parse_conjunctions(beliefs):
    beliefmap = {b.verb.object_side:b for b in beliefs}
   
    for b in beliefs:
        if not b.object: 
            continue
        root_token = b.verb.verb_span.root

        if root_token.head in beliefmap and             root_token.dep_ == "conj" and _spans_cconj(root_token):
          
            parent_object_span = beliefmap[root_token.head].object
            start,end = parent_object_span.start,parent_object_span.end
            end = min(b.object.end,end)
            if (b.subject.start > start):
                end = b.subject.start
            else:
                end = min(b.verb.verb_span.start,b.object.start)
            
            # Back out any final conjunction or punctuation
            while end > start and b.object.doc[end-1].pos_ in ["CCONJ","PUNCT"]:
                end-=1
            beliefmap[root_token.head].object = Span(b.object.doc,start,end)
    
    return beliefs


@Language.component('beliefs')
def extract_beliefs_doc(doc):
    noun_chunks = [nc for nc in doc.noun_chunks]
    for sent in doc.sents:
        if sent[-1].text =="?" or sent[0].lemma_ in ['who','what','how','why','where','when']:
            continue
        
        try:
            beliefs = extract_beliefs(sent)
        except Exception as e:
            print(f"Error {e} parsing {sent.text}")
            beliefs = []
        
        queue = set(beliefs)
        while queue:
            b = queue.pop()
            if b.nested_beliefs:
                queue.update(b.nested_beliefs)
            for nc in noun_chunks:
                if spans_intersect(b.subject,nc) and nc._.my_negex:
                    b.mark_negated(True)
        sent._.beliefs = beliefs
        doc._.beliefs += sent._.beliefs
    return doc


def add_to_pipe(nlp):
    if "my_negex" not in nlp.pipe_names:
        nlp.add_pipe("my_negex")
    nlp.add_pipe('coreferee')
    nlp.add_pipe('beliefs')
   
    # coreferee works with spacy==3.1.6 coreferee==1.1.3 en-core-web-lg==3.1.0 coreferee-model-en==1.0.0
    # setting up:
    # On tiamat install local tensorflow
    # pip install --ignore-installed --upgrade tensorflow-2.5.0-cp39-cp39-linux_x86_64.whl
    # pip install --user spacy==3.1.6
    # pip install --user coreferee
    # python -m spacy download en_core_web_lg
    # python -m coreferee install en


# In[9]:


#%run "belief_extraction_spacy.py"
#%run "preprocessing.py"
data = pd.read_csv("cleanedData1.csv")[['tweet_hash',"full_tweet_text"]]
data['cleaned'] = data['full_tweet_text'].apply(clean_tweet,args=(False,))
nlp = spacy.load("en_core_web_lg")
add_to_pipe(nlp)
def process_text(text):
    doc = nlp(text)
    result = []
    subjects = []
    for b in doc._.beliefs:
        cleaned = b.clean_belief()
        subject = b.clean_subject()
        if len(cleaned) > 0:
            result +=cleaned
            subjects+=subject
    return result,subjects

d['beliefs'] = d['cleaned'].apply(process_text)


# In[10]:


d.head(5)


# In[11]:


belief_dict = {}
belief_dict['belief'] = []
belief_dict['subject'] = []

for r in d['beliefs']:
    if len(r[0]) != 0:
        for b in r[0]:
            belief, subject = b[0], b[1]
            belief_dict['belief'].append(belief.lower())
            belief_dict['subject'].append(subject.lower())


# In[12]:


data = pd.DataFrame({'Beliefs':  belief_dict['belief'], 'Subjects':  belief_dict['subject']})


# In[13]:


data.shape


# In[14]:


data.head(5)


# In[15]:


data['Beliefs'][3]


# In[16]:


data['Subjects'][3]


# #### Code to merge similar subjects for example if two sentences have subjects that talk about 'climate' then the merged subject column will have name 'climate'

# #### Small implementation of knowledge-graph based method

# In[20]:


def merge_subjects(dataframe, threshold=70):
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


# In[21]:


merged_df = merge_subjects(data, threshold=70)


# In[23]:


merged_df.head(10)


# In[24]:


merged_df['Beliefs'][9]


# In[25]:


merged_df['Merged_Subject'][9]


# ### Merged Subjects

# In[26]:


merged_df.head(25)

