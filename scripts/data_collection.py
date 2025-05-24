import os
import json
import time
import random
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import spacy
from textblob import TextBlob
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import concurrent.futures
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import warnings
import hashlib
from difflib import SequenceMatcher
import jaro
from nltk.translate.bleu_score import sentence_bleu
import nltk.translate.bleu_score as bleu
from nltk.tokenize import word_tokenize
import nltk
from googletrans import Translator
from textaugment import EDA
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import PIL
from PIL import Image
import io
import base64
import requests
from urllib.parse import urlparse
import validators
import trafilatura
from newspaper import Article
import readability
import language_tool_python
import enchant
from autocorrect import Speller
import contractions
import emoji
import unicodedata
import ftfy
import re
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CADDataCollector:
    def __init__(self, output_dir: str = "data/raw", max_workers: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        
        # Create processed directory structure
        self.processed_dir = Path("data/processed")
        self.train_dir = self.processed_dir / "train"
        self.val_dir = self.processed_dir / "val"
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize NLP tools
        self.nlp = spacy.load('en_core_web_sm')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize BERT model for semantic similarity
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        
        # Initialize sentence transformer
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize text augmentation tools
        self.word_aug = naw.SynonymAug(aug_src='wordnet')
        self.char_aug = nac.KeyboardAug()
        self.sentence_aug = nas.ContextualWordEmbsAug(model_path='bert-base-uncased')
        
        # Initialize translation tool
        self.translator = Translator()
        
        # Initialize language tools
        self.language_tool = language_tool_python.LanguageTool('en-US')
        self.speller = Speller(lang='en')
        self.dictionary = enchant.Dict("en_US")
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(max_features=1000)
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
        # Initialize session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        
        # Initialize data validation rules
        self.validation_rules = {
            'min_description_length': 10,
            'max_description_length': 5000,
            'required_fields': ['title', 'description', 'cad_features'],
            'feature_thresholds': {
                'complexity': 0.0,
                'readability': 0.1,
                'technical_detail': 0.1
            },
            'quality_metrics': {
                'min_readability_score': 0.6,
                'min_technical_score': 0.4,
                'max_duplicate_similarity': 0.7,
                'min_metadata_completeness': 0.7,
                'min_semantic_coherence': 0.6,
                'min_domain_relevance': 0.6,
                'min_image_quality': 0.7,
                'min_cad_term_density': 0.3
            },
            'content_rules': {
                'min_unique_words': 8,
                'max_repeated_chars': 2,
                'min_sentence_count': 2,
                'max_sentence_length': 80,
                'min_paragraph_count': 2,
                'max_paragraph_length': 200,
                'min_technical_terms': 3,
                'max_spelling_errors': 2
            },
            'metadata_rules': {
                'required_tags': ['3D', 'CAD', 'model', 'design', 'engineering'],
                'min_tag_count': 3,
                'max_tag_count': 8,
                'required_categories': ['type', 'industry', 'material'],
                'min_metadata_fields': 8
            },
            'feature_rules': {
                'required_dimensions': ['length', 'width', 'height'],
                'min_feature_count': 5,
                'max_feature_count': 20,
                'required_technical_terms': 3,
                'min_complexity_score': 0.3
            },
            'cad_specific_rules': {
                'required_operations': ['extrude', 'revolve', 'sweep', 'loft'],
                'min_operation_count': 3,
                'max_operation_count': 20,
                'required_constraints': ['parallel', 'perpendicular', 'coincident'],
                'min_constraint_count': 2,
                'max_constraint_count': 15,
                'required_dimensions': ['length', 'width', 'height', 'radius', 'angle'],
                'min_dimension_count': 4,
                'max_dimension_count': 30,
                'required_features': ['holes', 'fillets', 'chamfers'],
                'min_feature_count': 2,
                'max_feature_count': 25,
                'material_properties': ['density', 'elasticity', 'strength'],
                'min_material_properties': 2,
                'max_material_properties': 10,
                'tolerance_specs': ['linear', 'angular', 'geometric'],
                'min_tolerance_count': 1,
                'max_tolerance_count': 8,
                'surface_finish': ['roughness', 'texture', 'coating'],
                'min_surface_properties': 1,
                'max_surface_properties': 5
            },
            'domain_specific_rules': {
                'mechanical': {
                    'required_features': ['bearing', 'shaft', 'gear', 'pulley', 'sprocket', 'coupling', 'clutch', 'brake'],
                    'min_feature_count': 2,
                    'max_feature_count': 15,
                    'required_operations': ['extrude', 'revolve', 'sweep', 'loft', 'fillet', 'chamfer'],
                    'min_operation_count': 3,
                    'max_operation_count': 20,
                    'required_constraints': ['parallel', 'perpendicular', 'coincident', 'concentric', 'tangent'],
                    'min_constraint_count': 3,
                    'max_constraint_count': 15,
                    'required_materials': ['steel', 'aluminum', 'brass', 'plastic'],
                    'min_material_count': 1,
                    'max_material_count': 5,
                    'required_tolerances': ['linear', 'angular', 'geometric', 'surface'],
                    'min_tolerance_count': 2,
                    'max_tolerance_count': 8,
                    'required_assemblies': ['rotating', 'sliding', 'fixed'],
                    'min_assembly_count': 1,
                    'max_assembly_count': 5,
                    'required_manufacturing': ['machining', 'casting', 'forging', 'welding'],
                    'min_manufacturing_count': 1,
                    'max_manufacturing_count': 4
                },
                'architectural': {
                    'required_features': ['wall', 'door', 'window', 'floor', 'ceiling', 'roof', 'stair', 'column', 'beam'],
                    'min_feature_count': 5,
                    'max_feature_count': 30,
                    'required_operations': ['extrude', 'sweep', 'loft', 'shell', 'pattern', 'mirror'],
                    'min_operation_count': 4,
                    'max_operation_count': 25,
                    'required_constraints': ['parallel', 'perpendicular', 'coincident', 'equal', 'symmetric'],
                    'min_constraint_count': 4,
                    'max_constraint_count': 20,
                    'required_materials': ['concrete', 'steel', 'wood', 'glass', 'brick'],
                    'min_material_count': 2,
                    'max_material_count': 8,
                    'required_tolerances': ['linear', 'angular', 'surface'],
                    'min_tolerance_count': 2,
                    'max_tolerance_count': 10,
                    'required_assemblies': ['structural', 'enclosure', 'interior'],
                    'min_assembly_count': 2,
                    'max_assembly_count': 6,
                    'required_manufacturing': ['precast', 'cast-in-place', 'prefabricated'],
                    'min_manufacturing_count': 1,
                    'max_manufacturing_count': 3
                },
                'electrical': {
                    'required_features': ['circuit', 'component', 'connection', 'terminal', 'switch', 'relay', 'transformer'],
                    'min_feature_count': 3,
                    'max_feature_count': 20,
                    'required_operations': ['extrude', 'sweep', 'pattern', 'mirror', 'array'],
                    'min_operation_count': 3,
                    'max_operation_count': 15,
                    'required_constraints': ['coincident', 'parallel', 'perpendicular', 'equal'],
                    'min_constraint_count': 2,
                    'max_constraint_count': 12,
                    'required_materials': ['copper', 'aluminum', 'plastic', 'ceramic'],
                    'min_material_count': 1,
                    'max_material_count': 6,
                    'required_tolerances': ['linear', 'angular', 'electrical'],
                    'min_tolerance_count': 1,
                    'max_tolerance_count': 6,
                    'required_assemblies': ['circuit', 'board', 'enclosure'],
                    'min_assembly_count': 1,
                    'max_assembly_count': 4,
                    'required_manufacturing': ['pcb', 'assembly', 'molding'],
                    'min_manufacturing_count': 1,
                    'max_manufacturing_count': 3
                },
                'automotive': {
                    'required_features': ['body', 'chassis', 'engine', 'transmission', 'suspension', 'brake', 'steering'],
                    'min_feature_count': 4,
                    'max_feature_count': 25,
                    'required_operations': ['extrude', 'revolve', 'sweep', 'loft', 'shell', 'pattern'],
                    'min_operation_count': 5,
                    'max_operation_count': 30,
                    'required_constraints': ['parallel', 'perpendicular', 'coincident', 'tangent', 'symmetric'],
                    'min_constraint_count': 4,
                    'max_constraint_count': 20,
                    'required_materials': ['steel', 'aluminum', 'plastic', 'rubber', 'glass'],
                    'min_material_count': 2,
                    'max_material_count': 8,
                    'required_tolerances': ['linear', 'angular', 'geometric', 'surface'],
                    'min_tolerance_count': 3,
                    'max_tolerance_count': 12,
                    'required_assemblies': ['body', 'chassis', 'powertrain', 'suspension'],
                    'min_assembly_count': 2,
                    'max_assembly_count': 6,
                    'required_manufacturing': ['stamping', 'casting', 'molding', 'welding'],
                    'min_manufacturing_count': 2,
                    'max_manufacturing_count': 5
                },
                'aerospace': {
                    'required_features': ['wing', 'fuselage', 'control', 'landing_gear', 'engine', 'propeller', 'rudder'],
                    'min_feature_count': 4,
                    'max_feature_count': 25,
                    'required_operations': ['extrude', 'sweep', 'loft', 'shell', 'pattern', 'mirror'],
                    'min_operation_count': 5,
                    'max_operation_count': 30,
                    'required_constraints': ['parallel', 'perpendicular', 'coincident', 'tangent', 'symmetric'],
                    'min_constraint_count': 4,
                    'max_constraint_count': 20,
                    'required_materials': ['aluminum', 'titanium', 'composite', 'steel'],
                    'min_material_count': 2,
                    'max_material_count': 8,
                    'required_tolerances': ['linear', 'angular', 'geometric', 'surface', 'aerodynamic'],
                    'min_tolerance_count': 3,
                    'max_tolerance_count': 12,
                    'required_assemblies': ['airframe', 'control_surface', 'landing_gear', 'propulsion'],
                    'min_assembly_count': 2,
                    'max_assembly_count': 6,
                    'required_manufacturing': ['machining', 'composite_layup', 'welding', 'riveting'],
                    'min_manufacturing_count': 2,
                    'max_manufacturing_count': 5
                },
                'marine': {
                    'required_features': ['hull', 'deck', 'keel', 'rudder', 'propeller', 'mast', 'sail'],
                    'min_feature_count': 4,
                    'max_feature_count': 25,
                    'required_operations': ['extrude', 'sweep', 'loft', 'shell', 'pattern', 'mirror'],
                    'min_operation_count': 5,
                    'max_operation_count': 30,
                    'required_constraints': ['parallel', 'perpendicular', 'coincident', 'tangent', 'symmetric'],
                    'min_constraint_count': 4,
                    'max_constraint_count': 20,
                    'required_materials': ['fiberglass', 'aluminum', 'steel', 'wood'],
                    'min_material_count': 2,
                    'max_material_count': 8,
                    'required_tolerances': ['linear', 'angular', 'geometric', 'surface', 'hydrodynamic'],
                    'min_tolerance_count': 3,
                    'max_tolerance_count': 12,
                    'required_assemblies': ['hull', 'deck', 'propulsion', 'steering'],
                    'min_assembly_count': 2,
                    'max_assembly_count': 6,
                    'required_manufacturing': ['molding', 'welding', 'laminating', 'assembly'],
                    'min_manufacturing_count': 2,
                    'max_manufacturing_count': 5
                },
                'industrial': {
                    'required_features': ['frame', 'support', 'bearing', 'gear', 'pulley', 'conveyor', 'hopper'],
                    'min_feature_count': 4,
                    'max_feature_count': 25,
                    'required_operations': ['extrude', 'revolve', 'sweep', 'loft', 'shell', 'pattern'],
                    'min_operation_count': 5,
                    'max_operation_count': 30,
                    'required_constraints': ['parallel', 'perpendicular', 'coincident', 'tangent', 'symmetric'],
                    'min_constraint_count': 4,
                    'max_constraint_count': 20,
                    'required_materials': ['steel', 'aluminum', 'plastic', 'rubber'],
                    'min_material_count': 2,
                    'max_material_count': 8,
                    'required_tolerances': ['linear', 'angular', 'geometric', 'surface'],
                    'min_tolerance_count': 3,
                    'max_tolerance_count': 12,
                    'required_assemblies': ['frame', 'drive', 'conveyor', 'hopper'],
                    'min_assembly_count': 2,
                    'max_assembly_count': 6,
                    'required_manufacturing': ['welding', 'machining', 'casting', 'assembly'],
                    'min_manufacturing_count': 2,
                    'max_manufacturing_count': 5
                }
            }
        }
        
        # Initialize additional NLP tools
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Initialize image processing tools
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.object_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        
        # Initialize CAD-specific tools
        self.cad_terms = set([
            'extrude', 'revolve', 'sweep', 'loft', 'fillet', 'chamfer',
            'shell', 'pattern', 'mirror', 'array', 'draft', 'hole',
            'thread', 'knurl', 'emboss', 'engrave', 'split', 'merge'
        ])
        self.cad_terms.update({
            # Advanced CAD operations
            'boolean', 'intersect', 'union', 'subtract', 'project', 'offset',
            'sweep', 'loft', 'revolve', 'extrude', 'fillet', 'chamfer',
            'shell', 'pattern', 'mirror', 'array', 'draft', 'hole',
            'thread', 'knurl', 'emboss', 'engrave', 'split', 'merge',
            'weld', 'trim', 'extend', 'scale', 'rotate', 'move',
            'copy', 'delete', 'hide', 'show', 'group', 'ungroup',
            'constrain', 'dimension', 'measure', 'analyze', 'simulate',
            'render', 'export', 'import', 'save', 'load', 'undo', 'redo'
        })

    def _extract_advanced_features(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract advanced features from text and metadata."""
        features = {
            'semantic_features': self._extract_semantic_features(text),
            'technical_features': self._extract_technical_features(text),
            'quality_features': self._extract_quality_features(text),
            'domain_features': self._extract_domain_features(text, metadata),
            'structural_features': self._extract_structural_features(text),
            'contextual_features': self._extract_contextual_features(text, metadata),
            'cad_specific_features': self._extract_cad_specific_features(text),
            'image_features': self._extract_image_features(metadata),
            'cross_modal_features': self._extract_cross_modal_features(text, metadata)
        }
        return features

    def _extract_semantic_features(self, text: str) -> Dict[str, Any]:
        """Extract semantic features using advanced NLP."""
        doc = self.nlp(text)
        
        features = {
            'entities': [ent.text for ent in doc.ents],
            'key_phrases': [chunk.text for chunk in doc.noun_chunks],
            'sentiment': TextBlob(text).sentiment.polarity,
            'topics': self._extract_topics(text),
            'semantic_roles': self._extract_semantic_roles(doc),
            'word_embeddings': self._get_word_embeddings(text),
            'semantic_graph': self._build_semantic_graph(doc)
        }
        return features

    def _extract_technical_features(self, text: str) -> Dict[str, Any]:
        """Extract technical features specific to CAD models."""
        features = {
            'dimensions': self._extract_dimensions(text),
            'materials': self._extract_materials(text),
            'operations': self._extract_cad_operations(text),
            'tolerances': self._extract_tolerances(text),
            'constraints': self._extract_constraints(text),
            'assembly_info': self._extract_assembly_info(text),
            'manufacturing_info': self._extract_manufacturing_info(text)
        }
        return features

    def _extract_quality_features(self, text: str) -> Dict[str, Any]:
        """Extract quality-related features."""
        features = {
            'readability': self._calculate_readability(text),
            'technical_detail': self._calculate_technical_detail(text),
            'completeness': self._calculate_completeness(text),
            'consistency': self._calculate_consistency(text),
            'clarity': self._calculate_clarity(text),
            'grammar_quality': self._check_grammar(text),
            'spelling_quality': self._check_spelling(text)
        }
        return features

    def _extract_domain_features(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract domain-specific features."""
        features = {
            'industry': self._identify_industry(text),
            'application': self._identify_application(text),
            'complexity_level': self._assess_complexity(text),
            'design_type': self._identify_design_type(text),
            'manufacturing_method': self._identify_manufacturing_method(text),
            'quality_grade': self._assess_quality_grade(text, metadata)
        }
        return features

    def _extract_structural_features(self, text: str) -> Dict[str, Any]:
        """Extract structural features of the text."""
        features = {
            'sentence_count': len(sent_tokenize(text)),
            'word_count': len(word_tokenize(text)),
            'avg_sentence_length': self._calculate_avg_sentence_length(text),
            'paragraph_count': len(text.split('\n\n')),
            'structure_score': self._calculate_structure_score(text),
            'formatting_quality': self._assess_formatting_quality(text)
        }
        return features

    def _extract_contextual_features(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contextual features considering metadata."""
        features = {
            'popularity_metrics': self._calculate_popularity_metrics(metadata),
            'user_engagement': self._calculate_user_engagement(metadata),
            'content_freshness': self._calculate_content_freshness(metadata),
            'source_reliability': self._assess_source_reliability(metadata),
            'content_relevance': self._calculate_content_relevance(text, metadata)
        }
        return features

    def _extract_cad_specific_features(self, text: str) -> Dict[str, Any]:
        """Extract CAD-specific features."""
        features = {
            'cad_operations': self._extract_cad_operations(text),
            'design_intent': self._extract_design_intent(text),
            'manufacturing_constraints': self._extract_manufacturing_constraints(text),
            'assembly_sequence': self._extract_assembly_sequence(text),
            'tolerance_specs': self._extract_tolerance_specs(text),
            'material_properties': self._extract_material_properties(text),
            'surface_finish': self._extract_surface_finish(text),
            'design_parameters': self._extract_design_parameters(text),
            'geometric_features': self._extract_geometric_features(text),
            'topological_features': self._extract_topological_features(text),
            'parametric_features': self._extract_parametric_features(text),
            'assembly_features': self._extract_assembly_features(text),
            'manufacturing_features': self._extract_manufacturing_features(text),
            'quality_features': self._extract_quality_features(text),
            'validation_features': self._extract_validation_features(text)
        }
        return features

    def _extract_geometric_features(self, text: str) -> Dict[str, Any]:
        """Extract geometric features."""
        features = {
            'primitives': self._extract_primitives(text),
            'surfaces': self._extract_surfaces(text),
            'curves': self._extract_curves(text),
            'vertices': self._extract_vertices(text),
            'edges': self._extract_edges(text),
            'faces': self._extract_faces(text),
            'volumes': self._extract_volumes(text),
            'transformations': self._extract_transformations(text)
        }
        return features

    def _extract_topological_features(self, text: str) -> Dict[str, Any]:
        """Extract topological features."""
        features = {
            'connectivity': self._extract_connectivity(text),
            'boundaries': self._extract_boundaries(text),
            'holes': self._extract_holes(text),
            'shells': self._extract_shells(text),
            'manifolds': self._extract_manifolds(text),
            'orientability': self._extract_orientability(text),
            'genus': self._extract_genus(text),
            'euler_characteristic': self._extract_euler_characteristic(text)
        }
        return features

    def _extract_parametric_features(self, text: str) -> Dict[str, Any]:
        """Extract parametric features."""
        features = {
            'parameters': self._extract_parameters(text),
            'constraints': self._extract_constraints(text),
            'equations': self._extract_equations(text),
            'variables': self._extract_variables(text),
            'functions': self._extract_functions(text),
            'dependencies': self._extract_dependencies(text),
            'driven_dimensions': self._extract_driven_dimensions(text),
            'reference_geometry': self._extract_reference_geometry(text)
        }
        return features

    def _extract_assembly_features(self, text: str) -> Dict[str, Any]:
        """Extract assembly features."""
        features = {
            'components': self._extract_components(text),
            'joints': self._extract_joints(text),
            'mates': self._extract_mates(text),
            'constraints': self._extract_assembly_constraints(text),
            'motion': self._extract_motion(text),
            'interference': self._extract_interference(text),
            'clearance': self._extract_clearance(text),
            'exploded_view': self._extract_exploded_view(text)
        }
        return features

    def _extract_manufacturing_features(self, text: str) -> Dict[str, Any]:
        """Extract manufacturing features."""
        features = {
            'processes': self._extract_processes(text),
            'tools': self._extract_tools(text),
            'fixtures': self._extract_fixtures(text),
            'operations': self._extract_manufacturing_operations(text),
            'tolerances': self._extract_manufacturing_tolerances(text),
            'surface_finish': self._extract_manufacturing_surface_finish(text),
            'materials': self._extract_manufacturing_materials(text),
            'cost_estimation': self._extract_cost_estimation(text)
        }
        return features

    def _extract_validation_features(self, text: str) -> Dict[str, Any]:
        """Extract validation features."""
        features = {
            'interference_check': self._extract_interference_check(text),
            'clearance_check': self._extract_clearance_check(text),
            'tolerance_check': self._extract_tolerance_check(text),
            'material_check': self._extract_material_check(text),
            'manufacturing_check': self._extract_manufacturing_check(text),
            'assembly_check': self._extract_assembly_check(text),
            'design_check': self._extract_design_check(text),
            'quality_check': self._extract_quality_check(text)
        }
        return features

    def _extract_image_features(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from associated images."""
        features = {
            'image_quality': self._assess_image_quality(metadata),
            'object_detection': self._detect_objects(metadata),
            'face_detection': self._detect_faces(metadata),
            'color_analysis': self._analyze_colors(metadata),
            'texture_analysis': self._analyze_textures(metadata),
            'composition_score': self._assess_composition(metadata)
        }
        return features

    def _extract_cross_modal_features(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features that combine text and image information."""
        features = {
            'text_image_alignment': self._calculate_text_image_alignment(text, metadata),
            'multimodal_coherence': self._calculate_multimodal_coherence(text, metadata),
            'cross_modal_similarity': self._calculate_cross_modal_similarity(text, metadata),
            'joint_embedding': self._generate_joint_embedding(text, metadata)
        }
        return features

    def _augment_data_advanced(self, model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate advanced augmented versions of the model data."""
        augmented = []
        
        # Original model
        augmented.append(model)
        
        # Text augmentation
        text_variations = self._generate_text_variations(model['description'])
        for text in text_variations:
            aug_model = model.copy()
            aug_model['description'] = text
            augmented.append(aug_model)
        
        # Feature augmentation
        feature_variations = self._generate_feature_variations_advanced(model['cad_features'])
        for features in feature_variations:
            aug_model = model.copy()
            aug_model['cad_features'] = features
            augmented.append(aug_model)
        
        # Metadata augmentation
        metadata_variations = self._generate_metadata_variations(model['metadata'])
        for metadata in metadata_variations:
            aug_model = model.copy()
            aug_model['metadata'] = metadata
            augmented.append(aug_model)
        
        # Cross-lingual augmentation
        cross_lingual_variations = self._generate_cross_lingual_variations(model)
        augmented.extend(cross_lingual_variations)
        
        # CAD-specific augmentation
        cad_variations = self._generate_cad_specific_variations(model)
        augmented.extend(cad_variations)
        
        # Image augmentation
        image_variations = self._generate_image_variations(model)
        augmented.extend(image_variations)
        
        # Multimodal augmentation
        multimodal_variations = self._generate_multimodal_variations(model)
        augmented.extend(multimodal_variations)
        
        return augmented

    def _generate_text_variations(self, text: str) -> List[str]:
        """Generate advanced text variations."""
        variations = []
        
        # Synonym replacement
        try:
            syn_variations = self.word_aug.augment(text, n=2)
            variations.extend(syn_variations)
        except Exception as e:
            logger.warning(f"Error in synonym augmentation: {e}")
        
        # Back-translation
        try:
            # Translate to another language and back
            for lang in ['es', 'fr', 'de']:
                translated = self.translator.translate(text, dest=lang).text
                back_translated = self.translator.translate(translated, dest='en').text
                variations.append(back_translated)
        except Exception as e:
            logger.warning(f"Error in back-translation: {e}")
        
        # Contextual augmentation
        try:
            ctx_variations = self.sentence_aug.augment(text, n=2)
            variations.extend(ctx_variations)
        except Exception as e:
            logger.warning(f"Error in contextual augmentation: {e}")
        
        return variations

    def _generate_feature_variations_advanced(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate advanced feature variations."""
        variations = []
        
        # Vary numerical features with controlled noise
        for feature, value in features.items():
            if isinstance(value, (int, float)):
                for noise_level in [0.05, 0.1, 0.15]:
                    new_features = features.copy()
                    noise = np.random.normal(0, noise_level)
                    new_features[feature] = value * (1 + noise)
                    variations.append(new_features)
        
        # Vary categorical features
        for feature, value in features.items():
            if isinstance(value, list):
                for _ in range(2):
                    new_features = features.copy()
                    new_features[feature] = random.sample(value, len(value))
                    variations.append(new_features)
        
        return variations

    def _generate_metadata_variations(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate metadata variations."""
        variations = []
        
        # Vary tags
        if 'tags' in metadata:
            for _ in range(2):
                new_metadata = metadata.copy()
                new_metadata['tags'] = self._generate_tag_variations(metadata['tags'])
                variations.append(new_metadata)
        
        # Vary timestamps
        if 'timestamp' in metadata:
            for _ in range(2):
                new_metadata = metadata.copy()
                new_metadata['timestamp'] = self._generate_timestamp_variation(metadata['timestamp'])
                variations.append(new_metadata)
        
        return variations

    def _generate_cross_lingual_variations(self, model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate cross-lingual variations."""
        variations = []
        
        # Translate to different languages and back
        languages = ['es', 'fr', 'de', 'it', 'pt']
        for lang in languages:
            try:
                # Translate title
                title_translated = self.translator.translate(model['title'], dest=lang).text
                title_back = self.translator.translate(title_translated, dest='en').text
                
                # Translate description
                desc_translated = self.translator.translate(model['description'], dest=lang).text
                desc_back = self.translator.translate(desc_translated, dest='en').text
                
                # Create new variation
                new_model = model.copy()
                new_model['title'] = title_back
                new_model['description'] = desc_back
                new_model['id'] = self._generate_model_id(title_back, model['source'])
                variations.append(new_model)
            except Exception as e:
                logger.warning(f"Error in cross-lingual augmentation for {lang}: {e}")
        
        return variations

    def _generate_cad_specific_variations(self, model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate CAD-specific variations."""
        variations = []
        
        # Vary CAD operations
        if 'cad_operations' in model['cad_features']:
            for _ in range(2):
                new_model = model.copy()
                new_model['cad_features']['cad_operations'] = self._vary_cad_operations(
                    model['cad_features']['cad_operations']
                )
                variations.append(new_model)
        
        # Vary design parameters
        if 'design_parameters' in model['cad_features']:
            for _ in range(2):
                new_model = model.copy()
                new_model['cad_features']['design_parameters'] = self._vary_design_parameters(
                    model['cad_features']['design_parameters']
                )
                variations.append(new_model)
        
        # Vary geometric features
        if 'geometric_features' in model['cad_features']:
            for _ in range(2):
                new_model = model.copy()
                new_model['cad_features']['geometric_features'] = self._vary_geometric_features(
                    model['cad_features']['geometric_features']
                )
                variations.append(new_model)
        
        # Vary topological features
        if 'topological_features' in model['cad_features']:
            for _ in range(2):
                new_model = model.copy()
                new_model['cad_features']['topological_features'] = self._vary_topological_features(
                    model['cad_features']['topological_features']
                )
                variations.append(new_model)
        
        # Vary parametric features
        if 'parametric_features' in model['cad_features']:
            for _ in range(2):
                new_model = model.copy()
                new_model['cad_features']['parametric_features'] = self._vary_parametric_features(
                    model['cad_features']['parametric_features']
                )
                variations.append(new_model)
        
        # Vary assembly features
        if 'assembly_features' in model['cad_features']:
            for _ in range(2):
                new_model = model.copy()
                new_model['cad_features']['assembly_features'] = self._vary_assembly_features(
                    model['cad_features']['assembly_features']
                )
                variations.append(new_model)
        
        # Vary manufacturing features
        if 'manufacturing_features' in model['cad_features']:
            for _ in range(2):
                new_model = model.copy()
                new_model['cad_features']['manufacturing_features'] = self._vary_manufacturing_features(
                    model['cad_features']['manufacturing_features']
                )
                variations.append(new_model)
        
        return variations

    def _generate_image_variations(self, model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate image variations."""
        variations = []
        
        if 'images' in model['metadata']:
            for image in model['metadata']['images']:
                # Apply various image transformations
                transformed_images = self._apply_image_transformations(image)
                for transformed in transformed_images:
                    new_model = model.copy()
                    new_model['metadata']['images'] = [transformed]
                    variations.append(new_model)
        
        return variations

    def _generate_multimodal_variations(self, model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multimodal variations."""
        variations = []
        
        # Combine text and image variations
        text_vars = self._generate_text_variations(model['description'])
        image_vars = self._generate_image_variations(model)
        
        for text in text_vars:
            for image_model in image_vars:
                new_model = model.copy()
                new_model['description'] = text
                new_model['metadata']['images'] = image_model['metadata']['images']
                variations.append(new_model)
        
        return variations

    def _validate_model_advanced(self, model: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Advanced validation of model data."""
        errors = []
        
        # Basic validation
        if not self._validate_model(model):
            errors.append("Basic validation failed")
            return False, errors
        
        # Content quality validation
        content_errors = self._validate_content_quality(model)
        errors.extend(content_errors)
        
        # Metadata validation
        metadata_errors = self._validate_metadata(model['metadata'])
        errors.extend(metadata_errors)
        
        # Feature validation
        feature_errors = self._validate_features(model['cad_features'])
        errors.extend(feature_errors)
        
        # Semantic validation
        semantic_errors = self._validate_semantics(model)
        errors.extend(semantic_errors)
        
        # CAD-specific validation
        cad_errors = self._validate_cad_specific(model)
        errors.extend(cad_errors)
        
        # Image validation
        image_errors = self._validate_images(model)
        errors.extend(image_errors)
        
        # Multimodal validation
        multimodal_errors = self._validate_multimodal(model)
        errors.extend(multimodal_errors)
        
        return len(errors) == 0, errors

    def _validate_cad_specific(self, model: Dict[str, Any]) -> List[str]:
        """Validate CAD-specific aspects."""
        errors = []
        
        # Check CAD operations
        if 'cad_operations' in model['cad_features']:
            ops = model['cad_features']['cad_operations']
            if not self._validate_cad_operations(ops):
                errors.append("Invalid CAD operations")
        
        # Check design parameters
        if 'design_parameters' in model['cad_features']:
            params = model['cad_features']['design_parameters']
            if not self._validate_design_parameters(params):
                errors.append("Invalid design parameters")
        
        # Check geometric features
        if 'geometric_features' in model['cad_features']:
            geom = model['cad_features']['geometric_features']
            if not self._validate_geometric_features(geom):
                errors.append("Invalid geometric features")
        
        # Check topological features
        if 'topological_features' in model['cad_features']:
            topo = model['cad_features']['topological_features']
            if not self._validate_topological_features(topo):
                errors.append("Invalid topological features")
        
        # Check parametric features
        if 'parametric_features' in model['cad_features']:
            param = model['cad_features']['parametric_features']
            if not self._validate_parametric_features(param):
                errors.append("Invalid parametric features")
        
        # Check assembly features
        if 'assembly_features' in model['cad_features']:
            assy = model['cad_features']['assembly_features']
            if not self._validate_assembly_features(assy):
                errors.append("Invalid assembly features")
        
        # Check manufacturing features
        if 'manufacturing_features' in model['cad_features']:
            mfg = model['cad_features']['manufacturing_features']
            if not self._validate_manufacturing_features(mfg):
                errors.append("Invalid manufacturing features")
        
        # Check quality features
        if 'quality_features' in model['cad_features']:
            qual = model['cad_features']['quality_features']
            if not self._validate_quality_features(qual):
                errors.append("Invalid quality features")
        
        # Check validation features
        if 'validation_features' in model['cad_features']:
            val = model['cad_features']['validation_features']
            if not self._validate_validation_features(val):
                errors.append("Invalid validation features")
        
        return errors

    def _validate_images(self, model: Dict[str, Any]) -> List[str]:
        """Validate image aspects."""
        errors = []
        
        if 'images' in model['metadata']:
            for image in model['metadata']['images']:
                if not self._validate_image_quality(image):
                    errors.append("Low image quality")
                if not self._validate_image_relevance(image, model):
                    errors.append("Image not relevant to model")
        
        return errors

    def _validate_multimodal(self, model: Dict[str, Any]) -> List[str]:
        """Validate multimodal aspects."""
        errors = []
        
        # Check text-image alignment
        alignment_score = self._calculate_text_image_alignment(
            model['description'],
            model['metadata']
        )
        if alignment_score < 0.6:
            errors.append(f"Low text-image alignment: {alignment_score}")
        
        # Check multimodal coherence
        coherence_score = self._calculate_multimodal_coherence(model)
        if coherence_score < 0.6:
            errors.append(f"Low multimodal coherence: {coherence_score}")
        
        return errors

    def _validate_domain_specific(self, model: Dict[str, Any], domain: str) -> List[str]:
        """Validate domain-specific aspects of the model."""
        errors = []
        
        if domain not in self.validation_rules['domain_specific_rules']:
            return errors
        
        rules = self.validation_rules['domain_specific_rules'][domain]
        
        # Validate required features
        if 'required_features' in rules:
            features = model['cad_features'].get('features', [])
            missing_features = [f for f in rules['required_features'] if f not in features]
            if missing_features:
                errors.append(f"Missing required features for {domain}: {missing_features}")
            
            feature_count = len(features)
            if feature_count < rules['min_feature_count']:
                errors.append(f"Insufficient features for {domain}: {feature_count} < {rules['min_feature_count']}")
            if feature_count > rules['max_feature_count']:
                errors.append(f"Too many features for {domain}: {feature_count} > {rules['max_feature_count']}")
        
        # Validate operations
        if 'required_operations' in rules:
            operations = model['cad_features'].get('operations', [])
            missing_operations = [op for op in rules['required_operations'] if op not in operations]
            if missing_operations:
                errors.append(f"Missing required operations for {domain}: {missing_operations}")
            
            operation_count = len(operations)
            if operation_count < rules['min_operation_count']:
                errors.append(f"Insufficient operations for {domain}: {operation_count} < {rules['min_operation_count']}")
            if operation_count > rules['max_operation_count']:
                errors.append(f"Too many operations for {domain}: {operation_count} > {rules['max_operation_count']}")
        
        # Validate constraints
        if 'required_constraints' in rules:
            constraints = model['cad_features'].get('constraints', [])
            missing_constraints = [c for c in rules['required_constraints'] if c not in constraints]
            if missing_constraints:
                errors.append(f"Missing required constraints for {domain}: {missing_constraints}")
            
            constraint_count = len(constraints)
            if constraint_count < rules['min_constraint_count']:
                errors.append(f"Insufficient constraints for {domain}: {constraint_count} < {rules['min_constraint_count']}")
            if constraint_count > rules['max_constraint_count']:
                errors.append(f"Too many constraints for {domain}: {constraint_count} > {rules['max_constraint_count']}")
        
        # Validate materials
        if 'required_materials' in rules:
            materials = model['cad_features'].get('materials', [])
            missing_materials = [m for m in rules['required_materials'] if m not in materials]
            if missing_materials:
                errors.append(f"Missing required materials for {domain}: {missing_materials}")
            
            material_count = len(materials)
            if material_count < rules['min_material_count']:
                errors.append(f"Insufficient materials for {domain}: {material_count} < {rules['min_material_count']}")
            if material_count > rules['max_material_count']:
                errors.append(f"Too many materials for {domain}: {material_count} > {rules['max_material_count']}")
        
        # Validate tolerances
        if 'required_tolerances' in rules:
            tolerances = model['cad_features'].get('tolerances', [])
            missing_tolerances = [t for t in rules['required_tolerances'] if t not in tolerances]
            if missing_tolerances:
                errors.append(f"Missing required tolerances for {domain}: {missing_tolerances}")
            
            tolerance_count = len(tolerances)
            if tolerance_count < rules['min_tolerance_count']:
                errors.append(f"Insufficient tolerances for {domain}: {tolerance_count} < {rules['min_tolerance_count']}")
            if tolerance_count > rules['max_tolerance_count']:
                errors.append(f"Too many tolerances for {domain}: {tolerance_count} > {rules['max_tolerance_count']}")
        
        # Validate assemblies
        if 'required_assemblies' in rules:
            assemblies = model['cad_features'].get('assemblies', [])
            missing_assemblies = [a for a in rules['required_assemblies'] if a not in assemblies]
            if missing_assemblies:
                errors.append(f"Missing required assemblies for {domain}: {missing_assemblies}")
            
            assembly_count = len(assemblies)
            if assembly_count < rules['min_assembly_count']:
                errors.append(f"Insufficient assemblies for {domain}: {assembly_count} < {rules['min_assembly_count']}")
            if assembly_count > rules['max_assembly_count']:
                errors.append(f"Too many assemblies for {domain}: {assembly_count} > {rules['max_assembly_count']}")
        
        # Validate manufacturing
        if 'required_manufacturing' in rules:
            manufacturing = model['cad_features'].get('manufacturing', [])
            missing_manufacturing = [m for m in rules['required_manufacturing'] if m not in manufacturing]
            if missing_manufacturing:
                errors.append(f"Missing required manufacturing methods for {domain}: {missing_manufacturing}")
            
            manufacturing_count = len(manufacturing)
            if manufacturing_count < rules['min_manufacturing_count']:
                errors.append(f"Insufficient manufacturing methods for {domain}: {manufacturing_count} < {rules['min_manufacturing_count']}")
            if manufacturing_count > rules['max_manufacturing_count']:
                errors.append(f"Too many manufacturing methods for {domain}: {manufacturing_count} > {rules['max_manufacturing_count']}")
        
        return errors

    def collect_data(self, num_pages: int = 20):
        """Collect data from all sources."""
        # Define sources with their specific configurations
        sources = {
            'grabcad': {
                'url': 'https://grabcad.com/library',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            'thingiverse': {
                'url': 'https://www.thingiverse.com/explore/newest',
                'card_selector': 'div.thing-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            'cults3d': {
                'url': 'https://cults3d.com/en/3d-models',
                'card_selector': 'div.product-card',
                'title_selector': 'h2',
                'description_selector': 'div.description'
            },
            'myminifactory': {
                'url': 'https://www.myminifactory.com/explore',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            'sketchfab': {
                'url': 'https://sketchfab.com/3d-models',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            'turbosquid': {
                'url': 'https://www.turbosquid.com/Search/3D-Models',
                'card_selector': 'div.product-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            'cgtrader': {
                'url': 'https://www.cgtrader.com/3d-models',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            'free3d': {
                'url': 'https://free3d.com/3d-models',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            'pinshape': {
                'url': 'https://pinshape.com/3d-models',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            'yeggi': {
                'url': 'https://www.yeggi.com/3d-models',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            'cadnav': {
                'url': 'https://www.cadnav.com/3d-models',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            '3dsky': {
                'url': 'https://3dsky.org/3d-models',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            '3dwarehouse': {
                'url': 'https://3dwarehouse.sketchup.com',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            'archdaily': {
                'url': 'https://www.archdaily.com/search/projects',
                'card_selector': 'div.project-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            'cadblocksfree': {
                'url': 'https://www.cadblocksfree.com',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            'bimobject': {
                'url': 'https://www.bimobject.com/en/products',
                'card_selector': 'div.product-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            '3dcontentcentral': {
                'url': 'https://www.3dcontentcentral.com',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            'traceparts': {
                'url': 'https://www.traceparts.com/en',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description'
            },
            'grabcad_mechanical': {
                'url': 'https://grabcad.com/library/mechanical',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description',
                'category': 'mechanical'
            },
            'grabcad_architectural': {
                'url': 'https://grabcad.com/library/architectural',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description',
                'category': 'architectural'
            },
            'grabcad_electrical': {
                'url': 'https://grabcad.com/library/electrical',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description',
                'category': 'electrical'
            },
            'grabcad_automotive': {
                'url': 'https://grabcad.com/library/automotive',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description',
                'category': 'automotive'
            },
            'grabcad_aerospace': {
                'url': 'https://grabcad.com/library/aerospace',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description',
                'category': 'aerospace'
            },
            'cadnav_mechanical': {
                'url': 'https://www.cadnav.com/3d-models/mechanical',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description',
                'category': 'mechanical'
            },
            'cadnav_architectural': {
                'url': 'https://www.cadnav.com/3d-models/architectural',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description',
                'category': 'architectural'
            },
            'cadnav_electrical': {
                'url': 'https://www.cadnav.com/3d-models/electrical',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description',
                'category': 'electrical'
            },
            'cadnav_automotive': {
                'url': 'https://www.cadnav.com/3d-models/automotive',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description',
                'category': 'automotive'
            },
            'cadnav_aerospace': {
                'url': 'https://www.cadnav.com/3d-models/aerospace',
                'card_selector': 'div.model-card',
                'title_selector': 'h3',
                'description_selector': 'div.description',
                'category': 'aerospace'
            }
        }
        
        # Collect data asynchronously
        all_models = []
        for source_name, config in sources.items():
            try:
                models = asyncio.run(self.scrape_source_async(source_name, config['url'], num_pages))
                all_models.extend(models)
                logger.info(f"Collected {len(models)} models from {source_name}")
            except Exception as e:
                logger.error(f"Error collecting from {source_name}: {e}")
                continue
        
        # Clean and validate data
        cleaned_models = []
        for model in all_models:
            is_valid, errors = self._validate_model_advanced(model)
            if is_valid:
                cleaned_models.append(model)
            else:
                logger.warning(f"Model validation failed: {errors}")
        
        logger.info(f"Cleaned data: {len(cleaned_models)} models after validation")
        
        # Remove duplicates
        unique_models = self._remove_duplicates(cleaned_models)
        logger.info(f"Unique models: {len(unique_models)} after duplicate removal")
        
        # Augment data
        augmented_models = []
        for model in unique_models:
            augmented_models.extend(self._augment_data_advanced(model))
        logger.info(f"Augmented data: {len(augmented_models)} models after augmentation")
        
        # Process and save the data
        self.process_and_save_data(augmented_models)
        
        logger.info("Data collection complete!")

def main():
    collector = CADDataCollector()
    collector.collect_data(num_pages=20)

if __name__ == "__main__":
    main() 