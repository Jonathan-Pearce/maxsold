# Create streamlit config
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
EOF

# Deploy to Streamlit Cloud
# 1. Push code to GitHub
git add dashboard/ .streamlit/
git commit -m "Add MaxSold analytics dashboard"
git push

# 2. Go to share.streamlit.io
# 3. Connect GitHub repo
# 4. Set main file: dashboard/maxsold_dashboard.py
# 5. Deploy!