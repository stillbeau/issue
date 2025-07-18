"""
Python Syntax Checker Tool
Helps identify syntax errors in your Python files
"""

import streamlit as st
import ast
import traceback

def check_python_syntax():
    """Check Python syntax for common issues"""
    
    st.header("🔍 Python Syntax Checker")
    st.markdown("Paste your Python code to check for syntax errors")
    
    # File content input
    st.subheader("📝 Code to Check")
    
    # Option 1: Paste code directly
    code_input = st.text_area(
        "Paste your Python code here:",
        height=300,
        placeholder="# Paste your data_processing.py content here..."
    )
    
    if st.button("🔍 Check Syntax") and code_input:
        check_syntax(code_input)
    
    # Option 2: Common syntax error patterns
    st.subheader("🔧 Common Syntax Error Fixes")
    
    common_fixes = {
        "Missing Quotes": {
            "error": "SyntaxError: EOL while scanning string literal",
            "fix": "Check for unclosed quotes: 'text or \"text",
            "example": "❌ name = 'John\n✅ name = 'John'"
        },
        "Missing Parentheses": {
            "error": "SyntaxError: unexpected EOF while parsing",
            "fix": "Check for unclosed parentheses: ( or [",
            "example": "❌ func(arg1, arg2\n✅ func(arg1, arg2)"
        },
        "Missing Colon": {
            "error": "SyntaxError: invalid syntax",
            "fix": "Check for missing colons after if, def, class, etc.",
            "example": "❌ if condition\n✅ if condition:"
        },
        "Indentation Error": {
            "error": "IndentationError: expected an indented block",
            "fix": "Check indentation - Python uses 4 spaces",
            "example": "❌ if True:\nprint('hello')\n✅ if True:\n    print('hello')"
        },
        "Mixed Tabs/Spaces": {
            "error": "IndentationError: inconsistent use of tabs and spaces",
            "fix": "Use only spaces (4 spaces per level)",
            "example": "Convert all tabs to 4 spaces"
        }
    }
    
    for error_type, info in common_fixes.items():
        with st.expander(f"🔧 {error_type}"):
            st.error(f"**Error**: {info['error']}")
            st.info(f"**Fix**: {info['fix']}")
            st.code(info['example'])

def check_syntax(code):
    """Check Python code for syntax errors"""
    
    try:
        # Try to parse the code
        ast.parse(code)
        st.success("✅ **No syntax errors found!**")
        st.info("Your Python code has valid syntax.")
        
    except SyntaxError as e:
        st.error("❌ **Syntax Error Found!**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Line Number", e.lineno if e.lineno else "Unknown")
            st.metric("Column", e.offset if e.offset else "Unknown")
        
        with col2:
            st.write(f"**Error Type**: {type(e).__name__}")
            st.write(f"**Message**: {e.msg}")
        
        # Show the problematic line
        if e.lineno:
            lines = code.split('\n')
            if e.lineno <= len(lines):
                st.subheader("🎯 Problematic Line:")
                
                # Show context (line before, problem line, line after)
                start_line = max(0, e.lineno - 2)
                end_line = min(len(lines), e.lineno + 1)
                
                for i in range(start_line, end_line):
                    line_num = i + 1
                    line_content = lines[i]
                    
                    if line_num == e.lineno:
                        st.error(f"Line {line_num}: {line_content}")
                        if e.offset:
                            # Show where the error is
                            pointer = " " * (e.offset - 1) + "^"
                            st.code(pointer)
                    else:
                        st.code(f"Line {line_num}: {line_content}")
        
        # Provide specific fix suggestions
        st.subheader("💡 Fix Suggestions:")
        
        error_msg = str(e.msg).lower()
        
        if "eol while scanning string literal" in error_msg:
            st.warning("🔧 **Missing Quote**: You have an unclosed string. Add a closing quote.")
        elif "unexpected eof while parsing" in error_msg:
            st.warning("🔧 **Missing Closing**: You're missing a closing ), ], or }.")
        elif "invalid syntax" in error_msg:
            st.warning("🔧 **Invalid Syntax**: Check for missing colons (:) after if/def/class statements.")
        else:
            st.warning(f"🔧 **General Fix**: {e.msg}")
    
    except IndentationError as e:
        st.error("❌ **Indentation Error Found!**")
        st.write(f"**Line {e.lineno}**: {e.msg}")
        st.warning("🔧 **Fix**: Check your indentation. Python uses 4 spaces per level.")
        
    except Exception as e:
        st.error(f"❌ **Other Error**: {e}")

def main():
    st.title("🐍 Python Syntax Checker")
    
    # Quick fix for data_processing.py
    st.subheader("⚡ Quick Fix for data_processing.py")
    
    st.info("""
    **Most likely issues in data_processing.py:**
    
    1. **Missing quotes** around strings
    2. **Unclosed parentheses** in function calls
    3. **Missing colons** after function definitions
    4. **Indentation errors** (mixed tabs/spaces)
    """)
    
    if st.button("📋 Show Clean data_processing.py"):
        st.success("✅ Use the clean version I provided above - it's syntax error free!")
        
        st.markdown("""
        **Steps to fix:**
        1. **Replace your current data_processing.py** with the clean version
        2. **Save the file**
        3. **Restart your Streamlit app**: `streamlit run app.py`
        """)
    
    # Main syntax checker
    check_python_syntax()

if __name__ == "__main__":
    main()
