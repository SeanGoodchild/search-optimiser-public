import streamlit as st

def inject_custom_styles():
    """Applies consistent styling to data tables throughout the app."""
    st.markdown("""
        <style>
        /* General table layout and font */
        .stDataFrame table {
            font-size: 0.95rem !important;
            border-collapse: collapse !important;
        }

        /* Cell padding and alignment */
        .stDataFrame table td, .stDataFrame table th {
            padding: 8px 16px !important;
            white-space: nowrap !important;
            vertical-align: middle !important;
        }

        /* Subtle hover feedback */
        .stDataFrame tbody tr:hover {
            background-color: rgba(60, 64, 67, 0.05) !important;
            transition: background-color 0.2s ease-in-out;
        }

        /* Incremental performance colors (semantic, accessible) */
        .inc-positive {
            color: #047857 !important;  /* green - good */
            font-weight: 600 !important;
        }

        .inc-negative {
            color: #b91c1c !important;  /* red - bad */
            font-weight: 600 !important;
        }

        .inc-neutral {
            color: #6b7280 !important;  /* gray - neutral */
            font-weight: 600 !important;
        }

        /* Keep the Incremental Δ column right-aligned for numeric clarity */
        .stDataFrame [data-testid="stTable"] td:nth-child(3),
        .stDataFrame [data-testid="stTable"] th:nth-child(3) {
            text-align: right !important;
        }
        </style>
    """, unsafe_allow_html=True)


def remove_top_padding():
    st.markdown(
        """
        <style>
        /* Remove the default top padding/margin Streamlit applies */
        .block-container
        {
            padding-top: 1rem;
            padding-left: 2rem;
            padding-bottom: 0rem;
            margin-top: 1rem;
        }
        
        section[data-testid="stSidebar"] {
            width: 230px !important; # Set the width to your desired value
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

def inject_logo(href: str | None = None, color: str = "#111", size_px: int = 36):
    link_open = f'<a href="{href}" target="_blank" rel="noopener">' if href else ""
    link_close = "</a>" if href else ""

    st.markdown(
        f"""
        <style>
        /* Position the logo in the top-right */
        .app-logo {{
            position: fixed;  /* use absolute if you want it to scroll with content */
            top: 1rem;
            left: 1.5rem;
            z-index: 10000;
            line-height: 0;   /* remove extra vertical whitespace */
        }}
        /* Size & color for the SVG */
        .app-logo svg {{
            width: {size_px}px;
            height: {int(size_px*24/37)}px; /* keep your 37:24 aspect ratio */
            display: block;
            color: {color};   /* used by fill: currentColor below */
        }}
        /* Make any .fill-current paths follow the SVG 'color' */
        .app-logo svg .fill-current {{
            fill: currentColor;
        }}
        </style>

        <div class="app-logo">
          {link_open}
          <svg viewBox="0 0 37 24" xmlns="http://www.w3.org/2000/svg" aria-label="Alli Logo">
            <path d="M11.0053 11.5456C9.69017 9.86255 7.89057 9.59326 6.85234 9.59326C3.14933 9.59326 0.0346375 12.3198 0.0346375 16.7293C0.0346375 20.4656 2.49178 23.9663 6.81774 23.9663C7.82136 23.9663 9.58635 23.7307 11.0053 22.115V23.5624H14.2584V9.96353H11.0053V11.5456ZM7.23303 21.1388C4.87971 21.1388 3.35697 19.1529 3.35697 16.7966C3.35697 14.3731 4.87971 12.4208 7.23303 12.4208C9.27488 12.4208 11.1783 13.8345 11.1783 16.8303C11.1783 19.6914 9.30949 21.1388 7.23303 21.1388Z" class="fill-current"></path>
            <path d="M21.6298 0H18.3767V23.5624H21.6298V0Z" class="fill-current"></path>
            <path d="M29.0012 0H25.7481V23.5624H29.0012V0Z" class="fill-current"></path>
            <path d="M36.3726 9.96362H33.1195V23.5625H36.3726V9.96362Z" class="fill-current"></path>
            <path d="M34.746 3.26514C33.5348 3.26514 32.6696 4.14031 32.6696 5.28477C32.6696 6.42923 33.5694 7.30441 34.746 7.30441C35.9573 7.30441 36.8571 6.42923 36.8571 5.28477C36.8571 4.14031 35.9227 3.26514 34.746 3.26514Z" class="fill-current"></path>
          </svg>
          {link_close}
        </div>
        """,
        unsafe_allow_html=True,
    )
