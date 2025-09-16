import streamlit as st
import torch
import requests
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv
import time
import pandas as pd
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai
import re
import json
from google import genai as google_genai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables at the beginning
load_dotenv()

class PubMedRetrieverTool:
    def forward(self, query: str) -> str:
        """Retrieve PubMed documents based on a query"""
        start_time = time.time()
        retrieved_docs = self._retrieve_pubmed_documents(query, max_results=20)
        retrieval_time = time.time() - start_time
        return f"Retrieved PubMed Documents (in {retrieval_time:.2f} seconds):\n{retrieved_docs}"
    
    def _retrieve_pubmed_documents(self, query: str, max_results: int = 20) -> str:
        """
        Searches PubMed using the E-utilities for relevant articles.
        Retrieves a maximum of 20 articles and returns a formatted string with
        each article's PMID, title, and abstract.
        """
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        # Step 1: Use esearch to get PubMed IDs
        esearch_url = base_url + "esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        
        try:
            response = requests.get(esearch_url, params=params, timeout=10)
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return "No relevant PubMed articles found."
            
            # Step 2: Use efetch to get article details (only abstract type)
            efetch_url = base_url + "efetch.fcgi"
            params_fetch = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
                "rettype": "abstract"
            }
            fetch_response = requests.get(efetch_url, params=params_fetch, timeout=15)
            xml_content = fetch_response.text
            root = ET.fromstring(xml_content)
            
            results = []
            for article in root.findall(".//PubmedArticle"):
                # Extract PMID (to be used as an identifier)
                pmid_elem = article.find(".//PMID")
                pmid_text = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else "No PMID"
                
                # Extract the title
                title_elem = article.find(".//ArticleTitle")
                title_text = ''.join(title_elem.itertext()).strip() if title_elem is not None else "No title"
                
                # Extract a simplified abstract
                abstract_elems = article.findall(".//Abstract/AbstractText")
                abstract_text = " ".join(''.join(abstract.itertext()).strip() 
                                        for abstract in abstract_elems if abstract.text) if abstract_elems else "No abstract available."
                
                results.append(
                    f"ID: {pmid_text}\nTitle: {title_text}\nAbstract: {abstract_text}\n"
                )
            
            return "\n".join(results)
        
        except requests.exceptions.Timeout:
            return "PubMed search timed out. Please try again with a more specific query."
        except Exception as e:
            return f"Error retrieving PubMed documents: {str(e)}"

class MedicalQueryPipeline:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
        # Initialize models and tools
        self.model_name = model_name
        self._initialize_models()
        self.pubmed_tool = PubMedRetrieverTool()
    
    def _initialize_models(self):
        """Initialize Gemini and the selected DeepSeek model"""
        # Configure Gemini API
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        
        # Load selected DeepSeek model
        config = AutoConfig.from_pretrained(self.model_name)
        self.eos_id = config.eos_token_id if config.eos_token_id is not None else 0
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model with low memory settings when necessary
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16,
                device_map="auto",
            )
        except Exception as e:
            st.warning(f"Loading model with reduced precision due to memory constraints: {str(e)}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            st.warning("Running on CPU. This will be slow for large models.")
    
    def extract_clean_answer(self, text):
        """Extract and clean the model's response"""
        # First try to find content between "Answer:" and "References:"
        answer_match = re.search(r'Answer:(.*?)(?:References:|$)', text, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # If that doesn't work, check if there's structured output with [Answer]
        if "[Answer]" in text:
            parts = re.split(r'\[Answer\](.*?)(?=\[|\Z)', text, re.DOTALL)
            if len(parts) > 1:
                return parts[1].strip()
        
        # Check for "Final Answer:" format
        if "Final Answer:" in text:
            parts = text.split("Final Answer:")
            if len(parts) > 1:
                return parts[1].strip()
                
        # Remove thinking or chat markup if present
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        
        # As a fallback, return the whole text but remove duplicative sections
        lines = text.splitlines()
        seen_chunks = set()
        unique_lines = []
        
        # Process in chunks to detect larger duplicate sections
        i = 0
        while i < len(lines):
            chunk = lines[i:i+3] if i+3 <= len(lines) else lines[i:]
            chunk_text = "\n".join(chunk)
            
            if chunk_text not in seen_chunks and chunk_text.strip():
                seen_chunks.add(chunk_text)
                unique_lines.extend(chunk)
            i += 3
        
        return "\n".join(unique_lines)
    
    def process_query(self, query, progress_bar=None, status_text=None):
        """Process a single query through the entire pipeline"""
        if status_text:
            status_text.text("Retrieving PubMed documents...")
        
        # Retrieve PubMed abstracts
        context_document = self.pubmed_tool.forward(query)
        
        # Update progress
        if progress_bar:
            progress_bar.progress(0.3)
        
        # Check if no results were found and refine the query using Gemini
        if "No relevant PubMed articles found." in context_document:
            if status_text:
                status_text.text("No results found. Refining query...")
            
            gemini_prompt = f"""
            input query: {query}
            modify this query into generic query to search on relevant pubmed articles to this query, only provide the modified query no text before or after the query
            """
            response = self.model_gemini.generate_content(gemini_prompt)
            refined_query = response.text.strip()
            
            if status_text:
                status_text.text(f"Searching with refined query: {refined_query}")
            
            context_document = self.pubmed_tool.forward(refined_query)
        
        # Extract clean context from the retriever response
        pattern = r"Retrieved PubMed Documents \(in .*\):\n(.*)"
        match = re.search(pattern, context_document, re.DOTALL)
        if match:
            clean_context = match.group(1).strip()
        else:
            clean_context = context_document
        
        # Update progress
        if progress_bar:
            progress_bar.progress(0.6)
            
        if status_text:
            status_text.text("Generating answer...")
        
        # Construct the prompt
        prompt = f"""
        You are a medical expert. Answer the following query using both your inherent knowledge and the provided PubMed abstract context.
        For any fact or statistic that you reference from the context, include a citation immediately after in the format:
            Citation: PMID: <ID>

        Query: {query}

        Context:
        {clean_context}
        You must provide a grounded responses with accurate citation from the context document.
        Provide a detailed, grounded final answer.
        """
     
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids, 
                attention_mask=attention_mask,  # Add this line
                max_new_tokens=2048, 
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id  # Add this line
            )


        raw_response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Clean the model response
        model_response = self.extract_clean_answer(raw_response)
        
        # Update progress
        if progress_bar:
            progress_bar.progress(1.0)
            
        if status_text:
            status_text.text("Done!")
        
        return {
            "query": query, 
            "context": clean_context, 
            "response": model_response,
            "raw_response": raw_response  # Store raw response for debugging
        }

# Facts evaluation class
class FactsEvaluator:
    def __init__(self):
        # Load evaluation prompts
        self.json_prompt = """You are a helpful and harmless AI assistant. You will be provided with a textual context and a model-generated response.\nYour task is to analyze the response sentence by sentence and classify each sentence according to its relationship with the provided context.\n\n**Instructions:**\n\n1. **Decompose the response into individual sentences.**\n2. **For each sentence, assign one of the following labels:**\n    * **`supported`**: The sentence is entailed by the given context.  Provide a supporting excerpt from the context. The supporting except must *fully* entail the sentence. If you need to cite multiple supporting excepts, simply concatenate them.\n    * **`unsupported`**: The sentence is not entailed by the given context. No excerpt is needed for this label.\n    * **`contradictory`**: The sentence is falsified by the given context. Provide a contradicting excerpt from the context.\n    * **`no_rad`**: The sentence does not require factual attribution (e.g., opinions, greetings, questions, disclaimers).  No excerpt is needed for this label.\n3. **For each label, provide a short rationale explaining your decision.**  The rationale should be separate from the excerpt.\n4. **Be very strict with your `supported` and `contradictory` decisions.** Unless you can find straightforward, indisputable evidence excerpts *in the context* that a sentence is `supported` or `contradictory`, consider it `unsupported`. You should not employ world knowledge unless it is truly trivial.\n\n**Input Format:**\n\nThe input will consist of two parts, clearly separated:\n\n* **Context:**  The textual context used to generate the response.\n* **Response:** The model-generated response to be analyzed.\n\n**Output Format:**\n\nFor each sentence in the response, output a JSON object with the following fields:\n\n* `"sentence"`: The sentence being analyzed.\n* `"label"`: One of `supported`, `unsupported`, `contradictory`, or `no_rad`.\n* `"rationale"`: A brief explanation for the assigned label.\n* `"excerpt"`:  A relevant excerpt from the context. Only required for `supported` and `contradictory` labels.\n\nOutput each JSON object on a new line.\n\n**Example:**\n\n**Input:**\n\n```\nContext: Apples are red fruits. Bananas are yellow fruits.\n\nResponse: Apples are red. Bananas are green. Bananas are cheaper than apples. Enjoy your fruit!\n```\n\n**Output:**\n\n{"sentence": "Apples are red.", "label": "supported", "rationale": "The context explicitly states that apples are red.", "excerpt": "Apples are red fruits."}\n{"sentence": "Bananas are green.", "label": "contradictory", "rationale": "The context states that bananas are yellow, not green.", "excerpt": "Bananas are yellow fruits."}\n{"sentence": "Bananas are cheaper than apples.", "label": "unsupported", "rationale": "The context does not mention the price of bananas or apples.", "excerpt": null}\n{"sentence": "Enjoy your fruit!", "label": "no_rad", "rationale": "This is a general expression and does not require factual attribution.", "excerpt": null}\n\n**Now, please analyze the following context and response:**\n\n**User Query:**\n{{user_request}}\n\n**Context:**\n{{context_document}}\n\n**Response:**\n{{response}}"""

        
        self.quality_prompt = """Your mission is to judge the response from an AI model, the *test* response, calibrating your judgement using a *baseline* response.\nPlease use the following rubric criteria to judge the responses:\n\n<START OF RUBRICS>\nYour task is to analyze the test response based on the criterion of "Instruction Following". Start your analysis with "Analysis".\n\n**Instruction Following**\nPlease first list the instructions in the user query.\nIn general, an instruction is VERY important if it is specifically asked for in the prompt and deviates from the norm. Please highlight such specific keywords.\nYou should also derive the task type from the user query and include the task-specific implied instructions.\nSometimes, no instruction is available in the user query.\nIt is your job to infer if the instruction is to autocomplete the user query or is asking the LLM for follow-ups.\nAfter listing the instructions, you should rank them in order of importance.\nAfter that, INDEPENDENTLY check if the test response and the baseline response meet each of the instructions.\nYou should itemize, for each instruction, whether the response meets, partially meets, or does not meet the requirement, using reasoning.\nYou should start reasoning first before reaching a conclusion about whether the response satisfies the requirement.\nCiting examples while reasoning is preferred.\n\nReflect on your answer and consider the possibility that you are wrong.\nIf you are wrong, explain clearly what needs to be clarified, improved, or changed in the rubric criteria and guidelines.\n\nIn the end, express your final verdict as one of the following three json objects:\n\n```json\n{{\n  "Instruction Following": "No Issues"\n}}\n```\n\n```json\n{{\n  "Instruction Following": "Minor Issue(s)"\n}}\n```\n\n```json\n{{\n  "Instruction Following": "Major Issue(s)"\n}}\n```\n\n<END OF RUBRICS>\n\n# Your task\n## User query\n<|begin_of_query|>\n{{full_prompt}}\n<|end_of_query|>\n\n## Test Response:\n<|begin_of_test_response|>\n{{response_a}}\n<|end_of_test_response|>\n\n## Baseline Response:\n<|begin_of_baseline_response|>\n{{response_b}}\n<|end_of_baseline_response|>\n\nPlease write your analysis and final verdict for the test response."""
        
        # Configure Gemini API
        google_api_key = Your_API_KEY
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=google_api_key)

    def generate_gemini(self, prompt):
        """Generate a response using Gemini model"""
        response = genai.GenerativeModel('gemini-1.5-flash-002').generate_content(prompt)
        return response.text

    def parse_structured_json(self, ans):
        """Parse JSON output from grounding evaluation"""
        if '```json' in ans:
            ans = ans.split('```json')[1].split('```')[0]
        ans = ans.strip()
        ans = ans.replace('}\n', '}\n@\n@\n')
        parsed_answers = []
        for line in ans.split('\n@\n@\n'):
            try:
                line = line.replace('\n', ' ')
                line = line.replace("\\'", "'")
                parsed = json.loads(line)
                parsed_answers.append(parsed)
            except:
                pass
        if len(parsed_answers) > 0:
            bool_ans = all(d['label'] == 'supported' or d['label'] == 'no_rad' for d in parsed_answers)
        else:
            bool_ans = False
        return bool_ans, parsed_answers

    def parse_json(self, ans):
        """Parse JSON output from quality evaluation"""
        parsed = {}
        if '```json' in ans:
            ans = ans.split('```json')[1]
            ans = ans.split('```')[0]
        ans = ans.replace('\n', ' ')
        try:
            parsed = json.loads(ans)
        except Exception as e:
            pass
        if 'Instruction Following' not in parsed:
            parsed['Instruction Following'] = 'Invalid'
        elif parsed['Instruction Following'] not in ['No Issues', 'Minor Issue(s)', 'Major Issue(s)', 'Invalid']:
            parsed['Instruction Following'] = 'Invalid'
        return parsed

    def evaluate_grounding(self, user_request, context_document, response):
        """Evaluate if response is grounded in the context"""
        prompt = self.json_prompt.replace('{{user_request}}', user_request).replace('{{context_document}}', context_document).replace('{{response}}', response)
        evaluation_text = self.generate_gemini(prompt)
        evaluation, parsed = self.parse_structured_json(evaluation_text)
        return evaluation, parsed

    def evaluate_quality(self, user_request, response_a, response_b):
        """Evaluate response quality against a reference"""
        prompt = self.quality_prompt.replace('{{user_request}}', user_request).replace('{{response_a}}', response_a).replace('{{response_b}}', response_b)
        evaluation_text = self.generate_gemini(prompt)
        parsed = self.parse_json(evaluation_text)
        return "No Issues" in parsed['Instruction Following'], parsed

    def evaluate_result(self, result):
        """Evaluate a single result for both grounding and quality"""
        # Generate a reference response using Gemini for quality evaluation
        reference_prompt = f"""
        You are a medical expert. Answer the following query based on your knowledge:
        
        Query: {result['query']}
        
        Provide a detailed, well-structured answer.
        """
        reference_response = self.generate_gemini(reference_prompt)
        
        # Evaluate grounding
        grounding_result, grounding_details = self.evaluate_grounding(
            user_request=result['query'],
            context_document=result['context'],
            response=result['response']
        )
        
        # Evaluate quality
        quality_result, quality_details = self.evaluate_quality(
            user_request=result['query'],
            response_a=result['response'],
            response_b=reference_response
        )
        
        # Calculate combined score
        combined_result = grounding_result and quality_result
        
        return {
            "grounding_evaluation": grounding_result,
            "quality_evaluation": quality_result,
            "combined_evaluation": combined_result,
            "grounding_details": grounding_details,
            "quality_details": quality_details
        }


# Streamlit app
def main():
    st.set_page_config(
        page_title="Medical RAG Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🩺 Medical Research Assistant")
    st.markdown("""
    This app uses Retrieval-Augmented Generation (RAG) with PubMed to answer medical queries. 
    The system retrieves relevant medical abstracts and provides answers with citations.
    """)
    
    # Sidebar for model selection
    st.sidebar.title("Model Settings")
    
    model_options = {
        "DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-QWen-7B": "deepseek-ai/DeepSeek-R1-Distill-QWen-7B",
        "DeepSeek-R1-Distill-QWen-14B": "deepseek-ai/DeepSeek-R1-Distill-QWen-14B"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys())
    )
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        show_raw_output = st.checkbox("Show Raw Model Output (for debugging)", value=False)
        max_abstracts = st.slider("Max PubMed Abstracts", min_value=5, max_value=30, value=20)
    
    # Initialize session state for storing results and evaluation
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    
    # Initialize session state
    if 'pipeline' not in st.session_state or st.session_state.model_name != model_options[selected_model]:
        st.session_state.model_name = model_options[selected_model]
        with st.spinner(f"Loading {selected_model}... This may take a moment."):
            st.session_state.pipeline = MedicalQueryPipeline(model_name=model_options[selected_model])
        st.success(f"Model loaded: {selected_model}")
    
    # Query input
    query = st.text_area("Enter your medical query:", height=100)
    
    # Example queries
    st.markdown("### Example Input Queries:")
    st.markdown("1. What are the latest treatments for type 2 diabetes?")
    st.markdown("2. How effective are mRNA vaccines against COVID-19 variants?")
    st.markdown("3. What's the connection between gut microbiome and depression?")
    
    # Process the query when the user clicks the button
    if st.button("Submit Query", type="primary") and query:
        st.markdown("---")
        
        # Set up progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process the query
        try:
            result = st.session_state.pipeline.process_query(
                query, 
                progress_bar=progress_bar, 
                status_text=status_text
            )
            
            # Store the current result in session state
            st.session_state.current_result = result
            
            # Reset evaluation results when a new query is processed
            st.session_state.evaluation_results = None
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
        
        finally:
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()

    # Display results section - moved outside of the Submit Query button
    if st.session_state.current_result is not None:
        result = st.session_state.current_result
        
        st.markdown("---")
        st.markdown("### Retrieved PubMed Abstracts")
        with st.expander("View PubMed References", expanded=False):
            st.markdown(result["context"])
        
        st.markdown("### Answer")
        st.markdown(result["response"])
        
        # Show raw output if enabled
        if show_raw_output:
            with st.expander("Raw Model Output (Debug)", expanded=False):
                st.text(result["raw_response"])
        
        # Add a download button for the results
        result_df = pd.DataFrame([{
            "Query": result["query"],
            "PubMed Context": result["context"],
            "Response": result["response"]
        }])
        
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="medical_query_results.csv",
            mime="text/csv",
        )
        
        # Add Facts Evaluation button section
        st.markdown("---")
        st.markdown("### Facts Evaluation")
        st.markdown("Evaluate how well the response is grounded in the PubMed abstracts and its overall quality.")
        
        if st.button("Evaluate Response", type="secondary"):
            with st.spinner("Evaluating response... This may take a moment."):
                try:
                    # Initialize the evaluator
                    evaluator = FactsEvaluator()
                    
                    # Evaluate the current result
                    evaluation_results = evaluator.evaluate_result(st.session_state.current_result)
                    
                    # Store evaluation results in session state
                    st.session_state.evaluation_results = evaluation_results
                    
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
                    st.exception(e)
        
        # Display evaluation results if available
        if st.session_state.evaluation_results is not None:
            eval_results = st.session_state.evaluation_results
            
            # Create columns for scores
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Grounding Score", 
                    value="True" if eval_results["grounding_evaluation"] else "False"
                )
            
            with col2:
                st.metric(
                    label="Quality Score", 
                    value="True" if eval_results["quality_evaluation"] else "False"
                )
            
            with col3:
                st.metric(
                    label="Combined Score", 
                    value="True" if eval_results["combined_evaluation"] else "False"
                )
            
            # Add download button for evaluation results
            eval_summary_df = pd.DataFrame([{
                "Query": st.session_state.current_result["query"],
                "Grounding Score": "True" if eval_results["grounding_evaluation"] else "False",
                "Quality Score": "True" if eval_results["quality_evaluation"] else "False",
                "Combined Score": "True" if eval_results["combined_evaluation"] else "False"
            }])
            
            csv = eval_summary_df.to_csv(index=False)
            st.download_button(
                label="Download Evaluation Results as CSV",
                data=csv,
                file_name="evaluation_results.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
