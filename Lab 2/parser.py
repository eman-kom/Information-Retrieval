from nltk.stem import PorterStemmer


class Parser:
    """
    Converts string query into reverse polish notation
    using shunting yard algorithm

    Attributes:
        porter (PorterStemmer): Used for stemming tokens in queries
        operators (dict): Store operator precedence
    """

    def __init__(self):
        self.porter = PorterStemmer()
        self.operators = {"(": -1, ")": -1, "NOT": 3, "AND": 2, "OR": 1}

    def parse(self, query: str) -> list:
        """
        Converts query into reverse polish notation using
        shunting yard algorithm

        Parameters:
            query (str): query taken from queries file

        Returns:
            reversePolish (list): query in reverse polish notation
        """
        spacedBrackets = self.splitBracket(query.strip())
        reversePolish = self.shuntingYard(spacedBrackets.split())
        return reversePolish

    def splitBracket(self, query: str) -> str:
        """
        Adds spaces between brackets for easier splitting

        Parameters:
            query (str): 1 query taken from queries file

        Returns:
            replaceRight (str): The same query with spaces between each bracket
        """
        replaceLeft = query.replace("(", " ( ")
        replaceRight = replaceLeft.replace(")", " ) ")
        return replaceRight

    def shuntingYard(self, query: list) -> list:
        """
        Performs the shunting yard algorithm to get the reverse polish
        notation of the query

        Parameters:
            query (list): The query that is splitted up by whitespaces

        Returns:
            queue (list): The query in reverse polish notation
        """
        queue = list()
        stack = list()

        while query:
            token = query.pop(0)

            if token not in self.operators:
                # also performs stemming and case-folding of non-operators
                queue.append(self.porter.stem(token.lower()))

            else:
                if token == "(":
                    stack.append(token)

                elif token == ")":
                    while (op := stack.pop()) != "(":
                        queue.append(op)

                else:
                    # checks for precedence
                    while stack and self.operators[stack[-1]] > self.operators[token]:
                        queue.append(stack.pop())

                    stack.append(token)

        # adds any remaining operators left from stack
        while stack:
            queue.append(stack.pop())

        return queue
