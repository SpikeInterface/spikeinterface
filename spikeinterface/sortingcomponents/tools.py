
def make_multi_method_doc(methods, ident='    '):
    doc = ""
    
    doc += "method: " + ', '.join(f"'{method.name}'" for method in methods) + '\n'
    doc += ident + "    Method to use.\n"

    for method  in methods:
        doc += "\n"
        doc += ident + f"arguments for method='{method.name}'"
        for line in method.params_doc.splitlines():
            doc += ident + line + '\n'

    return doc
