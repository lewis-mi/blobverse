
# ============================================================================
# BLOBVERSE FLASK API BACKEND
# Handles semantic embedding, clustering, and AI summarization
# ============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import numpy as np
import requests
from typing import List, Dict
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Initialize models (lazy loading)
_sentence_model = None
_umap_reducer = None

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer('all-mpnet-base-v2')
    return _sentence_model

# ============================================================================
# STEP 1: Anchor Importer - Fetch paper metadata
# ============================================================================

@app.route('/api/import-papers', methods=['POST'])
def import_papers():
    """
    Accept DOIs, URLs, or abstracts and retrieve metadata.
    Uses CrossRef API and Semantic Scholar API.
    """
    data = request.json
    identifiers = data.get('identifiers', [])  # List of DOIs or URLs

    papers = []
    for identifier in identifiers:
        if identifier.startswith('10.'):  # DOI
            paper = fetch_doi_metadata(identifier)
        elif 'arxiv.org' in identifier:
            paper = fetch_arxiv_metadata(identifier)
        else:
            paper = {'abstract': identifier, 'title': 'Custom Input'}

        if paper:
            papers.append(paper)

    return jsonify({
        'papers': papers,
        'total': len(papers),
        'timestamp': '2024-11-12T10:00:00Z'
    })

def fetch_doi_metadata(doi: str) -> Dict:
    """Fetch metadata from CrossRef API"""
    try:
        url = f'https://api.crossref.org/works/{doi}'
        response = requests.get(url)
        data = response.json()['message']

        return {
            'id': f'paper_{doi.replace("/", "_")}',
            'doi': doi,
            'title': data.get('title', [''])[0],
            'authors': [f"{a.get('given', '')} {a.get('family', '')}" 
                       for a in data.get('author', [])],
            'abstract': data.get('abstract', ''),
            'year': data.get('published-print', {}).get('date-parts', [[0]])[0][0],
            'citations': data.get('is-referenced-by-count', 0)
        }
    except Exception as e:
        print(f'Error fetching DOI {doi}: {e}')
        return None

def fetch_arxiv_metadata(url: str) -> Dict:
    """Fetch metadata from arXiv (simplified)"""
    # In production, parse arXiv ID and use arXiv API
    return {
        'id': 'paper_arxiv',
        'title': 'ArXiv Paper',
        'abstract': 'Retrieved from arXiv',
        'year': 2024
    }

# ============================================================================
# STEP 2: Semantic Embedding & Clustering
# ============================================================================

@app.route('/api/embed-and-cluster', methods=['POST'])
def embed_and_cluster():
    """
    Take paper abstracts, generate embeddings, reduce dimensions, 
    and cluster using UMAP + HDBSCAN.
    """
    data = request.json
    papers = data.get('papers', [])

    # Extract abstracts
    abstracts = [p.get('abstract', '') for p in papers]

    # Generate embeddings
    model = get_sentence_model()
    embeddings_768d = model.encode(abstracts)

    # Apply UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_neighbors=min(8, len(papers) - 1),
        n_components=3,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    embeddings_3d = reducer.fit_transform(embeddings_768d)

    # Apply HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, len(papers) // 5),
        min_samples=1,
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(embeddings_3d)

    # Prepare output
    result = {
        'embeddings': [],
        'clusters': {}
    }

    for i, paper in enumerate(papers):
        result['embeddings'].append({
            'paper_id': paper['id'],
            'vector_3d': embeddings_3d[i].tolist(),
            'cluster_id': int(cluster_labels[i])
        })

        # Group by cluster
        cluster_id = int(cluster_labels[i])
        if cluster_id not in result['clusters']:
            result['clusters'][cluster_id] = []
        result['clusters'][cluster_id].append(paper['id'])

    return jsonify(result)

# ============================================================================
# STEP 3: Visualization Prep - Convert to D3 format
# ============================================================================

@app.route('/api/prepare-visualization', methods=['POST'])
def prepare_visualization():
    """
    Convert embeddings to D3-ready node/link format with
    coordinates, colors, and force parameters.
    """
    data = request.json
    papers = data.get('papers', [])
    embeddings = data.get('embeddings', [])

    # Color palette for clusters
    colors = ['#C8FF61', '#FF6B9D', '#61D4FF', '#FFD700', '#FF6B35']

    # Create nodes
    nodes = []
    for i, paper in enumerate(papers):
        emb = embeddings[i]
        cluster_id = emb['cluster_id']

        # Normalize coordinates to canvas
        x = (emb['vector_3d'][0] + 5) * 40  # Scale to ~0-400
        y = (emb['vector_3d'][1] + 5) * 30

        nodes.append({
            'id': paper['id'],
            'label': paper['title'][:30],
            'full_title': paper['title'],
            'x': x,
            'y': y,
            'radius': 8 + (paper.get('citations', 0) / 100),
            'color': colors[cluster_id % len(colors)],
            'cluster_id': cluster_id,
            'year': paper.get('year', 2024),
            'abstract': paper.get('abstract', '')
        })

    # Create links based on cosine similarity
    links = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            # Calculate similarity between embeddings
            v1 = np.array(embeddings[i]['vector_3d'])
            v2 = np.array(embeddings[j]['vector_3d'])
            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            if similarity > 0.5:  # Threshold for link creation
                links.append({
                    'source': nodes[i]['id'],
                    'target': nodes[j]['id'],
                    'strength': float(similarity)
                })

    return jsonify({
        'nodes': nodes,
        'links': links,
        'metadata': {
            'total_nodes': len(nodes),
            'total_links': len(links),
            'year_range': [min(n['year'] for n in nodes), 
                          max(n['year'] for n in nodes)]
        }
    })

# ============================================================================
# STEP 5: AI Reflection Summary using Perplexity API
# ============================================================================

@app.route('/api/cluster-summary', methods=['POST'])
def cluster_summary():
    """
    Generate AI summary for a cluster using Perplexity API.
    Requires PERPLEXITY_API_KEY environment variable.
    """
    data = request.json
    cluster_id = data.get('cluster_id')
    abstracts = data.get('abstracts', '')

    # Construct prompt
    prompt = f'''Analyze these research paper abstracts from the same cluster:

{abstracts}

Answer these questions:
1. What unites these research papers?
2. What are the emerging questions or gaps in this research area?
3. What are the key concepts shared across these papers?

Provide a concise summary (3-4 sentences per question).'''

    # Call Perplexity API
    api_key = os.environ.get('PERPLEXITY_API_KEY')
    if not api_key:
        return jsonify({'error': 'Perplexity API key not configured'}), 500

    try:
        response = requests.post(
            'https://api.perplexity.ai/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'sonar-pro',
                'messages': [
                    {'role': 'system', 'content': 'You are a research analyst.'},
                    {'role': 'user', 'content': prompt}
                ]
            }
        )

        result = response.json()
        summary_text = result['choices'][0]['message']['content']

        return jsonify({
            'cluster_id': cluster_id,
            'unified_theme': summary_text.split('\n')[0],
            'emerging_questions': extract_questions(summary_text),
            'key_concepts': extract_concepts(summary_text)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def extract_questions(text: str) -> List[str]:
    """Extract question sentences from summary"""
    sentences = text.split('.')
    return [s.strip() + '?' for s in sentences if '?' in s][:3]

def extract_concepts(text: str) -> List[str]:
    """Extract key concepts (simplified)"""
    # In production, use NER or keyword extraction
    return ['semantic similarity', 'clustering', 'visualization']

# ============================================================================
# STEP 6: Export Constellation
# ============================================================================

@app.route('/api/export-constellation', methods=['POST'])
def export_constellation():
    """Generate markdown export of pinned constellation"""
    data = request.json
    pinned_nodes = data.get('pinned_nodes', [])
    clusters = data.get('clusters', {})

    markdown = "# My Research Constellation\n\n"
    markdown += f"*Generated: 2024-11-12*\n\n"
    markdown += f"## Overview\n"
    markdown += f"Total papers: {len(pinned_nodes)}\n\n"

    for cluster_id, papers in clusters.items():
        markdown += f"### Cluster {cluster_id}\n"
        for paper in papers:
            markdown += f"- {paper['title']} ({paper['year']})\n"
        markdown += "\n"

    return jsonify({
        'markdown': markdown,
        'filename': 'blobverse_constellation.md'
    })

# ============================================================================
# Health check
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'Blobverse API'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
