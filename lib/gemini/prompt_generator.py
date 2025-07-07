class PromptGenerator:
    """
    Generates prompts for Gemini fine-tune to predict next quarter mergers.
    """

    SYSTEM_TEMPLATE = """
        You are expert agent on predicting whether a company will engage in a merger in the next quarter.
        Given the name of the company, date of the filing, type of the filing and MD&A section of the filing,
        your task is to predict whether a company will engage in a merger in the next quarter.
        Analyze all the information given to you carefully to predict.
        If your prediction is a merger, the PREDICTION should be 1, otherwise it should always be 0.
        Also provide a reason for your prediction.
        Always respond with a JSON object in the following format:\n
        {{\n
        \"prediction\": PREDICTION\n
        \"reason\": REASON\n
        }}\n"
    """

    TEMPLATE = """
    <name_of_the_company>{company}</name_of_the_company>
    <filing_date>{filing_date}</filing_date>
    <mdna_section_of_report>{mdna_section}</mdna_section_of_report>
    """




    @classmethod
    def get_system_prompt(cls):
        return cls.SYSTEM_TEMPLATE
    @classmethod
    def generate_prompt(cls, company: str, filing_date: str, mdna_section: str) -> str:
        """
        Constructs a prompt with the given company, filing date, and MD&A text.

        Args:
            company (str): Name of the company.
            filing_date (str): Filing date in YYYY-MM-DD format.
            mdna_section (str): The MD&A section text.

        Returns:
            str: A formatted prompt string.
        """
        return cls.TEMPLATE.format(
            company=company,
            filing_date=filing_date,
            mdna_section=mdna_section
        )
