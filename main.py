from langchain_community.document_loaders import NewsURLLoader

def read_website_content(url):
    loader = NewsURLLoader([url])
    documents = loader.load()
    # print(f"Loaded {len(documents)} documents from {url}")
    return documents
    # for doc in documents:
    #     print(doc.page_content)

# Example usage
# if __name__ == "__main__":
#     url = "https://example.com"
#     read_website_content(url)
