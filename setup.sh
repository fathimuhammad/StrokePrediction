mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"fathimuhammad.2020@student.uny.ac.id\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
