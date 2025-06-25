# Dagshub + StreamLit 

## 1. What is Dagshub?  
Dagshub is a remote-collaboration platform **designed for machine-learning and data-science teams**. It merges three normally separate components into a single web interface:  
- **Git-style code versioning**  
- **DVC/LFS-based data & model versioning**  
- **MLflow-powered experiment tracking & visualization**  

In short, we can manage scripts the way we do on GitHub, store large datasets as easily as on Google Drive, and enjoy an MLflow-like dashboard that displays the performance of different models and hyper-parameter combinations at a glance.

Key strengths  
- Every dataset and model checkpoint is fully traceable.  
- Experiments can be filtered, sorted, and compared with interactive plots.  
- Each run automatically records its *code commit + data version + runtime environment*.

## 2. How our project connects to Dagshub  
Our group project uses **Python + Streamlit** to build a small web app that solves a business problem via linear regression. For experiment tracking we adopted the **Dagshub + MLflow** tech stack.

## Summary & Benefits  

By wiring MLflow logs into Dagshub we gained, at zero extra cost:  

- **Transparent experiment history** – no more “only screenshots, missing code/data” headaches.  
- **Synchronized data and code versions** – any result can be traced back to the exact CSV and commit used.  
- **Friendlier presentation** – reviewers open a web page to inspect parameters, curves, and download links, without configuring a local MLflow UI.  
- **Room to grow** – Dagshub also provides Issues, Pull Requests, and DVC Pipelines, giving the team a solid MLOps foundation for future iterations.  

Dagshub ensures our experiments not only **run**, but are also **tracked, shareable, and collaborative**.
