def text_to_sequence(self, text):
    
        sequence = []
        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = _curly_re.match(text)
            if not m:
                preprocessed_text = self._clean_text(text, [self.cleaner_names])
                sequence += self._symbols_to_sequence(preprocessed_text)
                break
            sequence += self._symbols_to_sequence(
                self._clean_text(m.group(1), [self.cleaner_names])
            )
            sequence += self._arpabet_to_sequence(m.group(2))
            text = m.group(3)

        # add eos tokens
        sequence += [self.eos_id]
        return sequence, preprocessed_text