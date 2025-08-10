## To Run

### For local development with `streamlit`

```
streamlit run brainswapchat/local_streamlit_app.py
```

If you want to create a requirements.txt file for pip install from the poetry .toml file automatically, I would recommend 

# Export with hashes for security
poetry export -f requirements.txt --output requirements.txt --without-hashes

Also consider this option, but it keeps the commit hashes:

# Export only production dependencies
poetry export -f requirements.txt --output requirements.txt --without dev

