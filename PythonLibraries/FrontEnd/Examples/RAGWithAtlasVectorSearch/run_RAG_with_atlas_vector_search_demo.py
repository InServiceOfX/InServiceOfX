from pathlib import Path
import sys

# Because of our need to have a monorepo structure that would use 3 different
# programming languages, C++, Python, Rust, and how Python imports modules,
# we need to make Python aware of from where to find modules.
number_of_parents_to_project_path = 3
current_filepath = Path(__file__).resolve() # Resolve to the absolute path.
project_path = \
    current_filepath.parents[number_of_parents_to_project_path].resolve()
sys.path.append(str(project_path.parent))

from InServiceOfX.CoreCode.CoreCode.MongoDBInterface.CreateURI import CreateURI
from InServiceOfX.CoreCode.CoreCode.MongoDBInterface.connect_to_client import (
    connect_to_client,
    CreateCustomCollection)
from InServiceOfX.CoreCode.CoreCode.Utilities.LoadEnvironmentFile import (
    load_environment_file,)
from InServiceOfX.VectorSearch.VectorSearch.WithMongoDB.AtlasVectorSearch import (
    AtlasVectorSearch,)

from InServiceOfX.FrontEnd.Examples.RAGWithAtlasVectorSearch.QuestionAndTwoOutputs import (
    QuestionAndTwoOutputs,)

# Below doesn't work apparently because this file isn't part of the package
# itself?
# from .QuestionAndTwoOutputs import QuestionAndTwoOutputs

if __name__ == '__main__':

    load_environment_file()    
    create_uri = CreateURI()
    new_uri = create_uri.prompt_password()
    client = connect_to_client(new_uri)

    create_custom_collection = CreateCustomCollection()
    collection = create_custom_collection.create_collection(client)

    atlas_vector_search = AtlasVectorSearch(collection=collection)

    gui = QuestionAndTwoOutputs(atlas_vector_search.query_data)
    gui.create_GUI()
    gui.run()