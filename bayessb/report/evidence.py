class Citation(object):
    """Simple class for storing citations.

    Parameters
    ----------
    citation : string
        The citation, as a string.
    pmid : string (optional)
        The PubMed ID.
    doi : string (optional)
        DOI for the citation.
    """
    def __init__(self, citation, pmid=None, doi=None):
        self.citation = citation
        self.pmid = pmid
        self.doi = doi

class Evidence(object):
    """Stores the specific evidence supporting an expectation for a model.

    Can include both text and images.

    Parameters
    ----------
    text : string
        The supporting text constituting the evidence.
    image : string (optional)
        A link/filename to data constituting the evidence.
    citation : :py:class:`Citation` (optional)
        A citation to the publication containing the evidence.
    """
    def __init__(self, text, image=None, citation=None):
        self.text = text
        self.image = image
        self.citation = citation

    def get_html(self):
        return self.text
