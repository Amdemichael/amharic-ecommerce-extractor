from ipywidgets import widgets, Layout
from IPython.display import display, HTML
from typing import List, Dict, Tuple
import spacy
from spacy.training import offsets_to_biluo_tags
import pandas as pd
from collections import defaultdict

class InteractiveLabeler:
    """Jupyter-based labeling interface"""
    
    def __init__(self, texts: List[str], labels: List[str] = None):
        self.texts = texts
        self.labels = labels or ["PRODUCT", "PRICE", "LOC"]
        self.current_idx = 0
        self.annotations = []
        self.entity_counts = defaultdict(int)
        
        # UI Elements
        self.text_display = widgets.Output(layout=Layout(width='100%'))
        self.entity_buttons = [widgets.Button(description=label) for label in self.labels]
        self.next_button = widgets.Button(description="Next Text")
        self.save_button = widgets.Button(description="Save Annotations")
        self.status = widgets.Output()
        
        # Setup callbacks
        for btn in self.entity_buttons:
            btn.on_click(self._on_entity_click)
        self.next_button.on_click(self._on_next)
        self.save_button.on_click(self._on_save)
        
        self._display_current()

    def _display_current(self):
        """Show current text and controls"""
        display(HTML(f"<h3>Text {self.current_idx+1}/{len(self.texts)}</h3>"))
        
        with self.text_display:
            self.text_display.clear_output()
            display(HTML(f"<div style='border:1px solid #ddd; padding:10px; margin:5px;'>{self.texts[self.current_idx]}</div>"))
        
        display(widgets.HBox(self.entity_buttons))
        display(widgets.HBox([self.next_button, self.save_button]))
        display(self.status)

    def _on_entity_click(self, btn):
        """Handle entity selection"""
        text_input = widgets.Text(
            description=f"Select text for {btn.description}:",
            layout=Layout(width='80%')
        )
        display(text_input)
        
        def save_annotation(b):
            text = text_input.value.strip()
            if text:
                self.annotations.append({
                    'text': text,
                    'label': btn.description,
                    'doc_idx': self.current_idx
                })
                self.entity_counts[btn.description] += 1
                with self.status:
                    self.status.clear_output()
                    print(f"✅ Added {btn.description}: '{text}'")
        
        save_btn = widgets.Button(description="Save Annotation")
        save_btn.on_click(save_annotation)
        display(save_btn)

    def _on_next(self, btn):
        """Move to next text"""
        if self.current_idx < len(self.texts) - 1:
            self.current_idx += 1
            self._display_current()
        else:
            with self.status:
                print("🎉 Finished all texts!")

    def _on_save(self, btn):
        """Save annotations to CONLL format"""
        nlp = spacy.blank("am")
        conll_lines = []
        
        for doc_idx, text in enumerate(self.texts):
            doc_ents = [e for e in self.annotations if e['doc_idx'] == doc_idx]
            offsets = [
                (text.find(e['text']), text.find(e['text']) + len(e['text']), e['label'])
                for e in doc_ents
            ]
            doc = nlp.make_doc(text)
            tags = offsets_to_biluo_tags(doc, offsets)
            
            for token, tag in zip(doc, tags):
                conll_lines.append(f"{token.text}\t{tag}\n")
            conll_lines.append("\n")
        
        with open("annotations.conll", "w", encoding="utf-8") as f:
            f.writelines(conll_lines)
        
        with self.status:
            print(f"💾 Saved {len(self.annotations)} annotations to annotations.conll")
            print("\nEntity Counts:")
            for label, count in self.entity_counts.items():
                print(f"- {label}: {count}")
