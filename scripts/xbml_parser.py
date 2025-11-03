import os

from PyPDF2 import PdfReader
try:
    from xbrl import XBRLParser, GAAP, GAAPSerializer
except ImportError as e:
    raise ImportError("Missing dependency 'python-xbrl'. Install it with: pip install python-xbrl") from e
from fpdf import FPDF


def extract_xbrl_from_pdf(pdf_path, xbrl_output="extracted_xbrl.xml"):
    """Extract XBRL data from PDF. Returns path to extracted XML or None."""
    try:
        xbrl_blocks = []
        with open(pdf_path, 'rb') as pdf_file:
            reader = PdfReader(pdf_file)
            print(f"  Processing PDF with {len(reader.pages)} pages...")
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        # Check for XBRL indicators
                        if 'xbrli:context' in text or '<xbrl' in text or 'xmlns:xbrl' in text:
                            xbrl_blocks.append(text)
                            print(f"    Found XBRL content on page {i+1}")
                except Exception as e:
                    print(f"    Warning: Could not extract text from page {i+1}: {e}")
        
        if not xbrl_blocks:
            print(f"  No XBRL found in {pdf_path}")
            # Try to extract as plain XML - sometimes XBRL is in XML format
            with open(pdf_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text and ('<?xml' in text or '<xbrl' in text.lower()):
                        xbrl_blocks.append(text)
            
            if not xbrl_blocks:
                return None
        
        with open(xbrl_output, 'w', encoding='utf-8') as f:
            f.writelines(xbrl_blocks)
        print(f"  XBRL extracted to {xbrl_output} ({len(xbrl_blocks)} blocks)")
        return xbrl_output
    except Exception as e:
        print(f"  Error extracting XBRL from {pdf_path}: {e}")
        return None


def parse_xml_fallback(xbrl_path):
    """Fallback XML parsing when XBRL parser fails."""
    try:
        import xml.etree.ElementTree as ET
        print(f"  Attempting XML fallback parsing...")
        tree = ET.parse(xbrl_path)
        root = tree.getroot()
        sentences = []
        count = 0
        for elem in root.iter():
            text = (elem.text or '').strip()
            if text and len(text) > 3:  # Filter very short text
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                sentences.append(f"{tag}: {text}")
                count += 1
                if count >= 1000:  # Limit to avoid huge outputs
                    break
        if sentences:
            print(f"  Generated {len(sentences)} sentences from XML fallback")
            return sentences
    except Exception as e:
        print(f"  XML fallback parsing failed: {e}")
    return []


def parse_xbrl_to_sentences(xbrl_path):
    """Parse XBRL XML file and convert to readable sentences."""
    try:
        print(f"  Parsing XBRL file: {xbrl_path}")
        
        # First validate the file exists and has content
        if not os.path.exists(xbrl_path):
            print(f"  Error: XBRL file does not exist: {xbrl_path}")
            return []
        
        # Try to parse with XBRLParser
        parser = XBRLParser()
        try:
            xbrl = parser.parse(xbrl_path)
            print(f"  XBRL parsed successfully")
        except (IndexError, ValueError, AttributeError) as e:
            print(f"  XBRL parser failed (file may not be valid XBRL): {e}")
            # Fall back to XML parsing
            return parse_xml_fallback(xbrl_path)
        except Exception as e:
            print(f"  XBRL parser error: {e}")
            # Fall back to XML parsing
            return parse_xml_fallback(xbrl_path)

        # Try using GAAP + GAAPSerializer first
        try:
            gaap_obj = GAAP(xbrl)
            if hasattr(gaap_obj, 'create_calculation_tree'):
                try:
                    gaap_obj.create_calculation_tree()
                except Exception as e:
                    print(f"    Warning: Could not create calculation tree: {e}")

            readable_facts = []
            try:
                serializer = GAAPSerializer()  # Some versions take no args
            except TypeError:
                serializer = GAAPSerializer  # Fallback in case it's a module-like object

            facts = None
            # Prefer to_list if available
            to_list_method = getattr(serializer, 'to_list', None)
            if callable(to_list_method):
                try:
                    facts = to_list_method(gaap_obj)
                    print(f"    Found {len(facts) if facts else 0} facts via to_list()")
                except Exception as e:
                    print(f"    to_list() failed: {e}")
                    facts = None
            # Try serialize or dump variants
            if facts is None:
                for method_name in ("serialize", "dump", "to_dict"):
                    method = getattr(serializer, method_name, None)
                    if callable(method):
                        try:
                            maybe = method(gaap_obj)
                            # Normalize into list of dict-like items
                            if isinstance(maybe, list):
                                facts = maybe
                                print(f"    Found {len(facts)} facts via {method_name}()")
                                break
                            if isinstance(maybe, dict) and 'facts' in maybe:
                                facts = maybe['facts']
                                print(f"    Found {len(facts)} facts via {method_name}()")
                                break
                        except Exception as e:
                            print(f"    {method_name}() failed: {e}")
                            continue

            if facts:
                for fact in facts:
                    try:
                        name = fact.get('name') or fact.get('concept') or 'Fact'
                        value = fact.get('value') or fact.get('val') or ''
                        context = fact.get('context') or fact.get('period') or ''
                        readable_facts.append(f"{name}: {value} (period: {context})")
                    except Exception as e:
                        print(f"    Error processing fact: {e}")
                        continue
                if readable_facts:
                    print(f"  Generated {len(readable_facts)} sentences from GAAP serializer")
                    return readable_facts
        except Exception as e:
            print(f"  GAAP parsing failed: {e}")

        # Fallback: naive XML parsing if serializer API differs
        sentences = parse_xml_fallback(xbrl_path)
        if sentences:
            return sentences
        
        print("  Warning: No sentences generated from XBRL parsing")
        return []
    except Exception as e:
        print(f"  Error parsing XBRL: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_sentences_to_pdf(sentences, pdf_path="parsed_xbrl_report.pdf"):
    """Save sentences to PDF. Returns True if successful, False if empty."""
    if not sentences:
        print(f"  Warning: No sentences to save, skipping PDF creation: {pdf_path}")
        return False
    
    try:
        print(f"  Creating PDF with {len(sentences)} sentences...")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=11)
        
        # Count how many sentences we actually write
        written_count = 0
        for i, sent in enumerate(sentences):
            try:
                # Clean up sentence for PDF - handle encoding issues
                sent_clean = str(sent).strip()
                
                # Remove or replace problematic characters
                sent_clean = sent_clean.encode('latin-1', errors='ignore').decode('latin-1')
                
                if sent_clean and len(sent_clean) > 0:
                    pdf.multi_cell(0, 10, sent_clean)
                    written_count += 1
                    
                    # Debug: show first few sentences
                    if written_count <= 3:
                        print(f"    Writing sentence {written_count}: {sent_clean[:50]}...")
                    
            except Exception as e:
                print(f"    Warning: Could not write sentence {i+1}: {e}")
                # Try to write a simplified version
                try:
                    simple_sent = str(sent)[:200].encode('ascii', errors='ignore').decode('ascii')
                    if simple_sent.strip():
                        pdf.multi_cell(0, 10, simple_sent)
                        written_count += 1
                except:
                    pass
        
        if written_count == 0:
            print(f"  Error: No sentences were written to PDF (all may have encoding issues)")
            # Create a placeholder page
            pdf.cell(0, 10, "No extractable content found", ln=True)
            pdf.output(pdf_path)
            return False
        
        pdf.output(pdf_path)
        print(f"  Saved PDF with {written_count}/{len(sentences)} sentences written: {pdf_path}")
        return True
    except Exception as e:
        print(f"  Error saving PDF: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_text_from_pdf_fallback(pdf_path):
    """Fallback: Extract regular text from PDF if XBRL extraction fails."""
    try:
        print(f"  Attempting fallback text extraction from PDF...")
        sentences = []
        with open(pdf_path, 'rb') as pdf_file:
            reader = PdfReader(pdf_file)
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        # Split into paragraphs/sentences
                        lines = text.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line and len(line) > 10:  # Filter very short lines
                                sentences.append(line)
                except Exception as e:
                    print(f"    Warning: Could not extract text from page {i+1}: {e}")
        
        if sentences:
            print(f"  Extracted {len(sentences)} text lines from PDF (fallback)")
            return sentences
        return []
    except Exception as e:
        print(f"  Fallback text extraction failed: {e}")
        return []


def process_pdf_to_natural_language(pdf_path, output_prefix="output"):
    """Process PDF to extract XBRL data and convert to natural language."""
    print(f"\nProcessing: {pdf_path}")
    print(f"Output prefix: {output_prefix}")
    
    # Try XBRL extraction first
    xbrl_path = extract_xbrl_from_pdf(pdf_path, f"{output_prefix}_xbrl.xml")
    sentences = []
    
    if xbrl_path:
        sentences = parse_xbrl_to_sentences(xbrl_path)
    
    # Fallback to regular text extraction if XBRL parsing fails or returns empty
    if not sentences:
        print("  XBRL parsing produced no results, trying fallback text extraction...")
        sentences = extract_text_from_pdf_fallback(pdf_path)
    
    # Debug: Show what we extracted
    print(f"  Extracted {len(sentences)} sentences/lines from PDF")
    if sentences:
        print(f"  First sentence preview: {str(sentences[0])[:100]}...")
        print(f"  Last sentence preview: {str(sentences[-1])[:100]}...")
    
    # Only proceed if we have content
    if not sentences:
        print(f"  Error: Could not extract any content from {pdf_path}")
        return
    
    # Save to text file
    try:
        txt_path = f"{output_prefix}_parsed.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for sent in sentences:
                f.write(str(sent) + '\n')
        print(f"  Saved text file with {len(sentences)} sentences: {txt_path}")
    except Exception as e:
        print(f"  Error saving text file: {e}")
        import traceback
        traceback.print_exc()
    
    # Save to PDF
    pdf_path_out = f"{output_prefix}_parsed.pdf"
    print(f"  Attempting to save PDF to: {pdf_path_out}")
    pdf_saved = save_sentences_to_pdf(sentences, pdf_path_out)
    
    if pdf_saved:
        print(f"✓ Successfully processed {pdf_path}")
    else:
        print(f"✗ Failed to create PDF for {pdf_path}")


# Batch processing: read from Annual Reports/pdfs and write to Annual Reports/parsed_pdfs
if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {_script_dir}")
    _project_root = os.path.abspath(os.path.join(_script_dir, os.pardir))
    _annual_reports_dir = os.path.join(_project_root, "data", "Annual Reports")
    print(f"Annual reports directory: {_annual_reports_dir}")
    _input_pdf_dir = os.path.join(_annual_reports_dir, "pdfs")
    _output_dir = os.path.join(_annual_reports_dir, "parsed_pdfs")
    
    print(f"\nInput PDF directory: {_input_pdf_dir}")
    print(f"Output directory: {_output_dir}\n")
    
    os.makedirs(_output_dir, exist_ok=True)
    
    if not os.path.isdir(_input_pdf_dir):
        print(f"✗ Error: Input directory not found: {_input_pdf_dir}")
        exit(1)
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(_input_pdf_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"✗ No PDF files found in {_input_pdf_dir}")
        exit(1)
    
    print(f"Found {len(pdf_files)} PDF file(s) to process\n")
    
    # Process each PDF
    successful = 0
    failed = 0
    
    for i, file in enumerate(pdf_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(pdf_files)}: {file}")
        print(f"{'='*60}")
        
        try:
            ticker = os.path.splitext(file)[0]
            input_pdf_path = os.path.join(_input_pdf_dir, file)
            output_prefix = os.path.join(_output_dir, ticker)
            
            process_pdf_to_natural_language(input_pdf_path, output_prefix=output_prefix)
            successful += 1
        except Exception as e:
            failed += 1
            print(f"✗ Error processing {file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(pdf_files)}")
    print(f"{'='*60}\n")
