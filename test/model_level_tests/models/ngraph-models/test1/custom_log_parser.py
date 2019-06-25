def custom_parse_logs(log_lines):
    outlines = log_lines.split('\n')
    test_line_idx = [
        idx for idx, line in enumerate(outlines) if line.startswith('TEST: ')
    ]
    return {
        "0":
        {outlines[i]: str(outlines[i + 1] == 'PASSED') for i in test_line_idx}
    }
