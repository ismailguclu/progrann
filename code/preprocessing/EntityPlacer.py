class EntityPlacer(object):

    def __init__(self, entities):
        self.entities = entities
        self.faulty = 0

    def __call__(self, doc):
        for start, end, label in self.entities:
            span = doc.char_span(start, end, label=label)
            print(span)
            if span is None:
                # self.faulty += 1
                # continue
                print(doc.char_span(start, end, label=label))
            doc.ents = list(doc.ents) + [span]
        return doc