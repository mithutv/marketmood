# ... (all your existing code up to the news section) ...

            # 8. News Sentiment Section
            st.markdown("### Recent Market News")
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            if news:
                for item in news[:3]:
                    st.markdown(f"""
                    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <div style="font-weight: bold;">{item.get('title')}</div>
                        <div style="font-size: 12px; color: #555;">Source: {item.get('publisher')} | <a href="{item.get('link')}" target="_blank">Read More</a></div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No recent news found for this ticker.")

            # 9. Broad Market Sentiment (Correctly indented inside the 'else' block)
            st.markdown("### Broad Market Sentiment (S&P 500)")
            spy = yf.Ticker("SPY")
            spy_news = spy.news

            for item in spy_news[:2]:
                st.markdown(f"- [{item.get('title')}]({item.get('link')})")
            
    # The 'except' must align vertically with the 'try' above
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
