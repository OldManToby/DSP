import graphviz

# Define your machine learning pipeline
pipeline = graphviz.Digraph()
pipeline.node('A', 'Data Preprocessing')
pipeline.node('B', 'Feature Engineering')
pipeline.node('C', 'Model Training')
pipeline.node('D', 'Model Evaluation')
pipeline.edges(['AB', 'BC', 'CD'])

# Specify the layout and render the diagram
pipeline.render('ml_pipeline_diagram', view=True)
