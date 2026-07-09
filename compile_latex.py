import re
import os

def compile_markdown_to_latex(md_path, tex_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split lines
    lines = content.split('\n')
    
    tex_lines = []
    
    # Headers metadata
    title = "Detecting AI-Generated User Reviews in Kazakh: A Study on BERT Model Performance and False Positive Reduction"
    authors = "Daulet Anekesh\\inst{1} \\and Irina Ualiyeva\\inst{1}"
    affiliation = "Al-Farabi Kazakh National University (KazNU), Almaty, Kazakhstan"
    emails = "\\email{\\{anekeshd, i.ualiyeva\\}@gmail.com}"
    
    # Boilerplate preamble
    tex_lines.append(r"\documentclass[runningheads]{llncs}")
    tex_lines.append(r"\usepackage{graphicx}")
    tex_lines.append(r"\usepackage{booktabs}")
    tex_lines.append(r"\usepackage{hyperref}")
    tex_lines.append(r"\usepackage[utf8]{inputenc}")
    tex_lines.append(r"\usepackage[T1,T2A]{fontenc}") # Set T2A (Cyrillic) as default active encoding
    tex_lines.append(r"\usepackage[english]{babel}") # Avoid kazakh.ldf missing error on Overleaf
    tex_lines.append(r"\usepackage{array}") # For custom column alignments
    tex_lines.append(r"\newcolumntype{P}[1]{>{\raggedright\arraybackslash}p{#1}}") # Force wrapping without hyphenation warnings
    tex_lines.append("")
    tex_lines.append(r"\begin{document}")
    tex_lines.append(r"\sloppy") # Relax word spacing to prevent Cyrillic/Model names from causing overfull hboxes
    tex_lines.append("")
    tex_lines.append(f"\\title{{{title}}}")
    tex_lines.append(r"\titlerunning{Detecting AI-Generated Reviews in Kazakh}")
    tex_lines.append("")
    tex_lines.append(f"\\author{{{authors}}}")
    tex_lines.append(r"\authorrunning{D. Anekesh and I. Ualiyeva}")
    tex_lines.append("")
    tex_lines.append(f"\\institute{{{affiliation} \\\\ {emails}}}")
    tex_lines.append("")
    tex_lines.append(r"\maketitle")
    tex_lines.append("")
    
    in_abstract = False
    in_ref = False
    in_table = False
    table_rows = []
    table_caption = "Table"
    table_count = 0
    in_itemize = False
    in_enumerate = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Clean blockquote prefix if present
        if line.strip().startswith('>'):
            line = re.sub(r'^\s*>\s*', '', line)
            
        # Detect table captions
        if '*Table ' in line or 'Table ' in line:
            clean_cap = line.strip('* \t\r\n')
            clean_cap = re.sub(r'^Table\s+\d+[:\s\-]*', '', clean_cap, flags=re.IGNORECASE)
            table_caption = clean_cap
            
        # Skip horizontal dividers and section separators
        if line.strip() == '---':
            i += 1
            continue
            
        # Skip title and authors metadata lines from markdown (only at the beginning of the document)
        if i < 10:
            if line.startswith('# ') and "Detecting" in line:
                i += 1
                continue
            if "Authors:" in line or "KazNU" in line or "Contact:" in line:
                i += 1
                continue
            
        # Abstract block
        if line.startswith('### Abstract'):
            in_abstract = True
            tex_lines.append(r"\begin{abstract}")
            i += 1
            continue
            
        if in_abstract:
            if line.startswith('**Keywords:**'):
                in_abstract = False
                tex_lines.append(r"\end{abstract}")
                # Format keywords
                kw_text = line.replace('**Keywords:**', '').strip()
                kw_list = [k.strip() for k in kw_text.split(',')]
                tex_lines.append(f"\\keywords{{{' \\and '.join(kw_list)}}}")
                tex_lines.append("")
                i += 1
                continue
            elif line.strip() == '---':
                in_abstract = False
                tex_lines.append(r"\end{abstract}")
                tex_lines.append("")
                i += 1
                continue
            else:
                # Add abstract text
                if line.strip():
                    tex_lines.append(line)
                i += 1
                continue

        # References block
        if line.startswith('## References'):
            in_ref = True
            tex_lines.append(r"\begin{thebibliography}{11}")
            i += 1
            continue
            
        if in_ref:
            if not line.strip():
                i += 1
                continue
            # Match bib entry like "1. Yeshpanov, R..."
            match = re.match(r'^\s*(\d+)\.\s*(.*)', line)
            if match:
                ref_num = match.group(1)
                ref_text = match.group(2)
                # Escape URL and percents in reference text
                ref_text = ref_text.replace('%', '\\%')
                ref_text = re.sub(r'(https?://[^\s]+)', r'\\url{\1}', ref_text)
                tex_lines.append(f"\\bibitem{{{ref_num}}}")
                tex_lines.append(ref_text)
                i += 1
                continue
            else:
                # End of references or normal text
                if line.startswith('## ') or line.startswith('# '):
                    in_ref = False
                    tex_lines.append(r"\end{thebibliography}")
                    tex_lines.append("")
                    continue
                else:
                    tex_lines.append(line)
                    i += 1
                    continue

        # Lists closing if blank line or non-list line
        is_list_item = line.strip().startswith(('* ', '- '))
        is_enum_item = re.match(r'^\s*\d+\.\s', line) is not None
        
        if in_itemize and not is_list_item and line.strip():
            tex_lines.append(r"\end{itemize}")
            tex_lines.append("")
            in_itemize = False
            
        if in_enumerate and not is_enum_item and line.strip() and not in_ref:
            tex_lines.append(r"\end{enumerate}")
            tex_lines.append("")
            in_enumerate = False

        # Table processing
        if in_table and (not line.startswith('|') or not line.strip()):
            in_table = False
            table_count += 1
            # Render LaTeX table
            tex_lines.append(r"\begin{table}")
            tex_lines.append(f"\\caption{{{table_caption}}}")
            tex_lines.append(f"\\label{{tab:table{table_count}}}")
            tex_lines.append(r"\centering")
            tex_lines.append(r"\footnotesize") # Use footnotesize to fit text width
            
            headers = [c.strip() for c in table_rows[0].split('|')[1:-1]]
            col_count = len(headers)
            
            if col_count == 7:
                tex_lines.append(r"\begin{tabular}{llccccc}")
            elif col_count == 6:
                if "Token Attribution" in table_caption or "Token" in headers[0]:
                    tex_lines.append(r"\begin{tabular}{lclclc}")
                else:
                    tex_lines.append(r"\begin{tabular}{lP{2.0cm}P{2.0cm}ccP{3.2cm}}")
            else:
                tex_lines.append(f"\\begin{{tabular}}{{{'c' * col_count}}}")
            tex_lines.append(r"\toprule")
            
            # Helper to parse columns with potential empty multicolumn spans
            def parse_row_cols(cols, is_header=False):
                formatted_cols = []
                j = 0
                while j < len(cols):
                    col = cols[j]
                    if col == "" and j > 0:
                        j += 1
                        continue
                    
                    # Count consecutive empty cells for multicolumn span
                    span = 1
                    while j + span < len(cols) and cols[j + span] == "":
                        span += 1
                    
                    c = col
                    c = c.replace('%', '\\%')
                    c = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', c)
                    c = re.sub(r'\*(.*?)\*', r'\\textit{\1}', c)
                    
                    if span > 1:
                        # Wrap in multicolumn
                        if is_header or c.startswith('\\textbf'):
                            formatted_cols.append(f"\\multicolumn{{{span}}}{{c}}{{{c}}}")
                        else:
                            formatted_cols.append(f"\\multicolumn{{{span}}}{{c}}{{\\textbf{{{c}}}}}")
                    else:
                        formatted_cols.append(c)
                    j += span
                return " & ".join(formatted_cols) + " \\\\"

            # Parse headers
            tex_lines.append(parse_row_cols(headers, is_header=True))
            tex_lines.append(r"\midrule")
            
            # Parse rows (skip index 1 as it is the divider | :--- |)
            for row_str in table_rows[2:]:
                cols = [c.strip() for c in row_str.split('|')[1:-1]]
                tex_lines.append(parse_row_cols(cols, is_header=False))
                
            tex_lines.append(r"\bottomrule")
            tex_lines.append(r"\end{tabular}")
            tex_lines.append(r"\end{table}")
            tex_lines.append("")
            table_rows = []
            if not line.strip():
                i += 1
                continue
                
        if line.startswith('|'):
            in_table = True
            table_rows.append(line)
            i += 1
            continue

        # Headings
        if line.startswith('## '):
            heading_text = line[3:].strip()
            # Strip number prefix (e.g. "1. Introduction" -> "Introduction")
            heading_text = re.sub(r'^\d+\.\s*', '', heading_text)
            heading_text = heading_text.replace('&', '\\&')
            tex_lines.append(f"\\section{{{heading_text}}}")
            i += 1
            continue
        elif line.startswith('### '):
            heading_text = line[4:].strip()
            heading_text = re.sub(r'^\d+\.\d+\s*', '', heading_text)
            heading_text = heading_text.replace('&', '\\&')
            tex_lines.append(f"\\subsection{{{heading_text}}}")
            i += 1
            continue
        elif line.startswith('#### '):
            heading_text = line[5:].strip()
            heading_text = re.sub(r'^\d+\.\s*', '', heading_text)
            heading_text = heading_text.rstrip('. \t\r\n')
            heading_text = heading_text.replace('&', '\\&')
            tex_lines.append(f"\\subsubsection{{{heading_text}.}}")
            i += 1
            continue

        # Images
        match_img = re.match(r'^!\[(.*?)\]\((.*?)\)', line)
        if match_img:
            alt_text = match_img.group(1)
            img_path = match_img.group(2)
            tex_lines.append(r"\begin{figure}")
            tex_lines.append(r"\centering")
            tex_lines.append(f"\\includegraphics[width=\\textwidth]{{{img_path}}}")
            tex_lines.append(f"\\caption{{{alt_text}}}")
            tex_lines.append(r"\label{fig:dotplot}")
            tex_lines.append(r"\end{figure}")
            tex_lines.append("")
            i += 1
            continue

        # List Items (Itemize)
        if is_list_item:
            if not in_itemize:
                tex_lines.append(r"\begin{itemize}")
                in_itemize = True
            clean_line = line.strip()[2:]
            tex_lines.append(f"\\item {clean_line}")
            i += 1
            continue

        # Enumerated Items
        if is_enum_item and not in_ref:
            if not in_enumerate:
                tex_lines.append(r"\begin{enumerate}")
                in_enumerate = True
            clean_line = re.sub(r'^\s*\d+\.\s*', '', line)
            tex_lines.append(f"\\item {clean_line}")
            i += 1
            continue

        # Paragraph
        if line.strip():
            tex_lines.append(line)
            
        i += 1

    if in_ref:
        tex_lines.append(r"\end{thebibliography}")
    if in_itemize:
        tex_lines.append(r"\end{itemize}")
    if in_enumerate:
        tex_lines.append(r"\end{enumerate}")
        
    tex_lines.append("")
    tex_lines.append(r"\end{document}")
    
    # Process text formatting on all lines in output (bold, italic, citations, percentages)
    tex_content = "\n".join(tex_lines)
    
    # Convert markdown bold **text** to \textbf{text}
    tex_content = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', tex_content)
    
    # Convert markdown italic *text* to \textit{text}
    tex_content = re.sub(r'\*(.*?)\*', r'\\textit{\1}', tex_content)
    
    # Convert citation markers [1] to \cite{1}
    # Be careful not to replace things like [1] in bibliography
    # We only match [num] if it's not inside \bibitem{num} or \begin{thebibliography}
    # So we do a negative lookbehind for bibitem
    tex_content = re.sub(r'(?<!\\bibitem\{)(?<!\\begin\{thebibliography\}\{)(?<!\\cite\{)\[(\d+)\]', r'\\cite{\1}', tex_content)
    
    # Escape percent signs (%) in body text unless already escaped
    # We match % only if not preceded by \
    tex_content = re.sub(r'(?<!\\)%', r'\\%', tex_content)
    
    # Replace en-dash/em-dash with ---
    tex_content = tex_content.replace('—', '---')
    
    # Replace quotes: "text" -> ``text''
    # Simple regex to replace double quotes in pairs
    parts = tex_content.split('"')
    new_parts = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1: # Odd index means inside quote
            new_parts.append(f"``{part}''")
        else:
            new_parts.append(part)
    tex_content = "".join(new_parts)
    
    # Replace quotes inside brackets/parentheses or isolated
    # e.g., *"жазбаларыңыздан"* -> \textit{``жазбаларыңыздан''}
    # We already converted *text* to \textit{text}, which results in \textit{"жазбаларыңыздан"}
    # The double quote splitter above handles this if they are double quotes.
    
    # Write to tex_path
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)
        
    print(f"Successfully compiled LaTeX document to: {tex_path}")

if __name__ == '__main__':
    compile_markdown_to_latex('paper_draft.md', 'paper.tex')
