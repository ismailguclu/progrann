class EntityPlacer(object):

    def __init__(self, entities):
        self.entities = entities
        self.faulty = 0

    def __call__(self, doc):
        for start, end, label, e_text in self.entities:
            span = doc.char_span(start, end, label=label)
            if span is None:
                count_leading_spaces = len(e_text) - len(e_text.lstrip(" "))
                count_ending_spaces = len(e_text) - len(e_text.rstrip(" "))
                if count_leading_spaces > 0:
                    new_start = start - count_leading_spaces
                    span = doc.char_span(new_start, end, label=label)
                elif count_ending_spaces > 0:
                    new_end = end + count_ending_spaces
                    span = doc.char_span(start, new_end, label=label)
                else:
                    print(doc.char_span(start, end, label=label))
            doc.ents = list(doc.ents) + [span]
        return doc