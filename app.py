from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import os
from google import genai
import json
import re
import time
from typing import Dict, List, Optional, Set, Tuple
from paper_content_fetcher import PaperContentFetcher

app = Flask(__name__)
CORS(app)

# Set API key environment variable for Gemini
os.environ['GEMINI_API_KEY'] = 'YOUR_API_KEY_HERE'

class SemanticScholarAPI:
    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {}
    
    def get_paper_details(self, paper_id: str) -> Optional[Dict]:
        """Get paper details from Semantic Scholar"""
        # Handle different ID formats
        if paper_id.startswith("10.48550/arXiv."):
            # Convert DOI format to ArXiv ID
            arxiv_id = paper_id.replace("10.48550/arXiv.", "")
            paper_id = f"arXiv:{arxiv_id}"
            print(f"  - Converted DOI to ArXiv ID: {paper_id}")
        elif paper_id.startswith("arXiv:"):
            # Already in correct format
            pass
        elif "arXiv" in paper_id.lower() or bool(re.match(r'^\d{4}\.\d{4,5}', paper_id)):
            # Try to extract ArXiv ID
            match = re.search(r'(\d{4}\.\d{4,5})', paper_id)
            if match:
                paper_id = f"arXiv:{match.group(1)}"
                print(f"  - Extracted ArXiv ID: {paper_id}")
        
        url = f"{self.base_url}/paper/{paper_id}"
        fields = [
            "paperId", "externalIds", "title", "abstract", "year",
            "authors", "venue", "citationCount", "referenceCount",
            "openAccessPdf"
        ]
        params = {"fields": ",".join(fields)}
        
        try:
            print(f"  - Fetching paper details from: {url}")
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                paper = response.json()
                print(f"  - Found paper: {paper.get('title', 'Unknown')[:60]}...")
                return paper
            elif response.status_code == 429:
                # Rate limit hit - wait and retry
                print(f"  - Rate limit hit, waiting 2 seconds and retrying...")
                time.sleep(2)
                response = requests.get(url, params=params, headers=self.headers)
                if response.status_code == 200:
                    paper = response.json()
                    print(f"  - Found paper: {paper.get('title', 'Unknown')[:60]}...")
                    return paper
                else:
                    print(f"  - Error fetching paper after retry: {response.status_code} - {response.text}")
                    return None
            else:
                print(f"  - Error fetching paper: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"  - Request failed: {e}")
            return None
    
    def get_paper_references(self, paper_id: str) -> List[Dict]:
        """Get all references of a paper"""
        url = f"{self.base_url}/paper/{paper_id}/references"
        fields = [
            "paperId", "externalIds", "title", "abstract", "year",
            "authors", "contexts", "intents"
        ]
        params = {
            "fields": ",".join(fields),
            "limit": 100  # Limit for faster processing
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            elif response.status_code == 429:
                # Rate limit hit - wait and retry
                print(f"    - Rate limit hit on references, waiting 2 seconds...")
                time.sleep(2)
                response = requests.get(url, params=params, headers=self.headers)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("data", [])
                else:
                    print(f"    - Error fetching references after retry: {response.status_code}")
                    return []
            else:
                print(f"    - Error fetching references: {response.status_code}")
                return []
        except Exception as e:
            print(f"    - Request failed: {e}")
            return []

def extract_dependencies_with_gemini(paper_details: Dict, paper_content: Optional[str], references: List[Dict]) -> List[Dict]:
    """Use Gemini to identify which references are actual dependencies with detailed reasoning"""
    
    paper_title = paper_details.get("title", "Unknown")
    paper_abstract = paper_details.get("abstract", "")
    
    # Prepare reference list for Gemini
    ref_list = []
    ref_map = {}
    for i, ref in enumerate(references):
        cited_paper = ref.get("citedPaper", {})
        if cited_paper:
            ref_id = cited_paper.get("paperId")
            title = cited_paper.get("title", "Unknown")
            authors = cited_paper.get("authors", [])
            author_names = ", ".join([a.get("name", "Unknown") for a in authors[:2]])
            if len(authors) > 2:
                author_names += " et al."
            year = cited_paper.get("year", "N/A")
            
            contexts = ref.get("contexts", [])
            context_text = " | ".join(contexts[:3]) if contexts else "No context available"
            
            ref_list.append(f"{i+1}. [{ref_id}] {title} ({author_names}, {year})")
            ref_list.append(f"   Citation contexts: {context_text}")
            ref_map[i+1] = {
                "paper_id": ref_id,
                "title": title,
                "authors": author_names,
                "year": year
            }
    
    if not ref_list:
        return []
    
    # Use full paper content if available, otherwise fall back to abstract
    content_to_analyze = paper_content if paper_content else f"Title: {paper_title}\n\nAbstract: {paper_abstract}"
    
    # Truncate content if too long (Gemini has token limits)
    max_content_length = 800000  # Characters
    if len(content_to_analyze) > max_content_length:
        content_to_analyze = content_to_analyze[:max_content_length] + "\n\n[Content truncated due to length...]"
    
    # Create prompt for Gemini
    prompt = f"""You are analyzing a mathematics paper to identify its true dependencies. A true dependency is a reference whose mathematical results (theorems, lemmas, or definitions) are directly used in proving or establishing the results of the analyzed paper.

PAPER CONTENT:
{content_to_analyze}

REFERENCES:
{chr(10).join(ref_list)}

TASK:
Analyze each reference and determine if it's a true dependency. For each reference, provide:
1. Whether it's a dependency (true/false)
2. A specific reason explaining your decision
3. If it's a dependency, list the specific mathematical elements (theorems, lemmas, definitions) that are used

OUTPUT FORMAT:
Return a JSON array with the following structure:
{{
  "dependencies": [
    {{
      "reference_number": 1,
      "paper_id": "abc123",
      "is_dependency": true,
      "reason": "The paper directly uses Theorem 3.2 and Lemma 4.1 from this reference to prove the main result in Section 5",
      "specific_elements": ["Theorem 3.2", "Lemma 4.1", "Definition 2.1"]
    }},
    {{
      "reference_number": 2,
      "paper_id": "def456",
      "is_dependency": false,
      "reason": "Only mentioned in the introduction for historical context and motivation",
      "specific_elements": []
    }}
  ]
}}

IMPORTANT:
- Only mark as dependency if mathematical results are DIRECTLY USED in proofs or definitions
- Background mentions, comparisons, and motivational citations are NOT dependencies
- Look for phrases like "by Theorem X in [Y]", "using the result from", "applying Lemma", "follows from"
- Be conservative: when in doubt, it's likely NOT a dependency
"""
    
    try:
        # Create client inside the function to ensure API key is set
        client = genai.Client()
        # Try with retries
        for attempt in range(3):
            try:
                # Use Gemini 2.5 Pro
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt
                )
                break
            except Exception as e:
                if "503" in str(e) or "overloaded" in str(e).lower():
                    if attempt < 2:
                        print(f"    - Gemini overloaded, waiting 3 seconds and retrying...")
                        time.sleep(3)
                        continue
                raise
        
        # Parse JSON response
        result_text = response.text.strip()
        
        # Extract JSON from the response (sometimes Gemini adds markdown formatting)
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group()
        
        result = json.loads(result_text)
        dependencies = result.get("dependencies", [])
        
        # Process and return the results
        processed_dependencies = []
        for dep in dependencies:
            if dep.get("is_dependency", False):
                ref_num = dep.get("reference_number")
                if ref_num in ref_map:
                    processed_dependencies.append({
                        "paper_id": ref_map[ref_num]["paper_id"],
                        "title": ref_map[ref_num]["title"],
                        "authors": ref_map[ref_num]["authors"],
                        "year": ref_map[ref_num]["year"],
                        "reason": dep.get("reason", ""),
                        "specific_elements": dep.get("specific_elements", [])
                    })
        
        return processed_dependencies
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response text: {response.text[:500]}...")
        # Fallback to simple extraction
        return _fallback_dependency_extraction(references)
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback: use papers with methodology/result intents
        return _fallback_dependency_extraction(references)

def _fallback_dependency_extraction(references: List[Dict]) -> List[Dict]:
    """Fallback method using citation intents"""
    dependencies = []
    for ref in references[:5]:  # Limit to 5 for performance
        intents = ref.get("intents", [])
        if "methodology" in intents or "result" in intents:
            cited_paper = ref.get("citedPaper", {})
            if cited_paper and cited_paper.get("paperId"):
                authors = cited_paper.get("authors", [])
                author_names = ", ".join([a.get("name", "Unknown") for a in authors[:2]])
                if len(authors) > 2:
                    author_names += " et al."
                    
                dependencies.append({
                    "paper_id": cited_paper["paperId"],
                    "title": cited_paper.get("title", "Unknown"),
                    "authors": author_names,
                    "year": cited_paper.get("year", "N/A"),
                    "reason": f"Citation intent indicates {', '.join(intents)}",
                    "specific_elements": []
                })
    return dependencies

def build_dependency_graph(doi: str, max_depth: int = 2) -> Dict:
    """Build the dependency graph starting from a DOI"""
    api = SemanticScholarAPI()
    content_fetcher = PaperContentFetcher()
    
    nodes = []
    edges = []
    processed = set()
    node_id_counter = 0
    paper_id_to_node_id = {}
    dependency_reasons = {}  # Store reasons for dependencies
    
    def process_paper(paper_id: str, depth: int, parent_node_id: Optional[int] = None):
        nonlocal node_id_counter
        
        if paper_id in processed or depth > max_depth:
            return
        
        processed.add(paper_id)
        
        # Get paper details
        paper = api.get_paper_details(paper_id)
        if not paper:
            return
        
        # Create node
        current_node_id = node_id_counter
        paper_id_to_node_id[paper_id] = current_node_id
        node_id_counter += 1
        
        # Extract author names
        authors = paper.get("authors", [])
        author_names = ", ".join([a.get("name", "Unknown") for a in authors[:3]])
        if len(authors) > 3:
            author_names += " et al."
        
        # Create short label for graph
        title = paper.get("title", "Unknown")
        short_title = title[:40] + "..." if len(title) > 40 else title
        
        # Fetch paper content if available
        print(f"Processing paper: {title[:60]}... (Level {depth})")
        paper_content, content_source = content_fetcher.fetch_paper_content(paper)
        has_full_text = paper_content is not None
        print(f"  - Full text: {'Yes' if has_full_text else 'No'} ({content_source})")
        
        nodes.append({
            "id": current_node_id,
            "label": short_title,
            "title": title,
            "year": paper.get("year", "N/A"),
            "authors": author_names,
            "level": depth,
            "has_full_text": has_full_text,
            "content_source": content_source
        })
        
        # Add edge from parent with dependency reason if available
        if parent_node_id is not None:
            edge_key = f"{parent_node_id}-{current_node_id}"
            edge_data = {
                "from": parent_node_id,
                "to": current_node_id
            }
            if edge_key in dependency_reasons:
                edge_data["title"] = dependency_reasons[edge_key]["reason"]
                edge_data["label"] = ", ".join(dependency_reasons[edge_key]["specific_elements"][:2])
            edges.append(edge_data)
        
        # Get references and find dependencies
        if depth < max_depth:
            # Use the paper's actual Semantic Scholar ID for fetching references
            ss_paper_id = paper.get('paperId')
            references = api.get_paper_references(ss_paper_id)
            
            if references:
                # Use Gemini to identify true dependencies with full paper content
                print(f"  - Found {len(references)} references, analyzing dependencies...")
                dependencies = extract_dependencies_with_gemini(
                    paper,
                    paper_content,
                    references
                )
                print(f"  - Identified {len(dependencies)} true dependencies")
                
                # Process each dependency
                for dep in dependencies[:5]:  # Limit to 5 per level for performance
                    dep_id = dep["paper_id"]
                    
                    # Store dependency reason for edge
                    if dep_id not in processed:  # Only store if we'll actually process this paper
                        future_node_id = node_id_counter + len([p for p in dependencies[:5] if p["paper_id"] not in processed and dependencies.index(p) < dependencies.index(dep)])
                        edge_key = f"{current_node_id}-{future_node_id}"
                        dependency_reasons[edge_key] = {
                            "reason": dep["reason"],
                            "specific_elements": dep["specific_elements"]
                        }
                    
                    process_paper(dep_id, depth + 1, current_node_id)
    
    # Start processing from the given DOI
    process_paper(doi, 0)
    
    return {
        "nodes": nodes,
        "edges": edges
    }

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/analyze_paper', methods=['POST'])
def analyze_paper():
    try:
        data = request.json
        doi = data.get('doi', '').strip()
        
        if not doi:
            return jsonify({"error": "DOI is required"}), 400
        
        print(f"\n=== Starting analysis for DOI: {doi} ===")
        
        # Build dependency graph
        graph_data = build_dependency_graph(doi)
        
        if not graph_data["nodes"]:
            # More specific error message
            api = SemanticScholarAPI()
            paper = api.get_paper_details(doi)
            if not paper:
                error_msg = f"Paper with identifier '{doi}' not found in Semantic Scholar. Please check the DOI/ID and try again."
                print(f"Error: {error_msg}")
                return jsonify({"error": error_msg}), 404
            else:
                error_msg = "Paper found but no dependencies could be identified. This might be because the paper has no references or they couldn't be analyzed."
                print(f"Error: {error_msg}")
                return jsonify({"error": error_msg}), 404
        
        print(f"=== Analysis complete: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges ===\n")
        return jsonify(graph_data)
        
    except Exception as e:
        import traceback
        print(f"Error in analyze_paper: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)