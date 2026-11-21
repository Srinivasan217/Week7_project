
questions = [
    # 6 exact-term questions
    {
        "type": "exact",
        "question": "What is the title of document FINANCE-001?",
        "expected": "Payroll Processing Policy",
        "filename": "finance_01.pdf"
    },
    {
        "type": "exact",
        "question": "What is the title of document FINANCE-002?",
        "expected": "Travel Reimbursement",
        "filename": "finance_02.docx"
    },
    {
        "type": "exact",
        "question": "What is the title of document HR-001?",
        "expected": "Employee Attendance Policy",
        "filename": "hr_01.pdf"
    },
    {
        "type": "exact",
        "question": "What is the title of document HR-002?",
        "expected": "Annual Leave Policy",
        "filename": "hr_02.docx"
    },
    {
        "type": "exact",
        "question": "What is the title of document IT-002?",
        "expected": "Password Security Policy",
        "filename": "it_02.docx"
    },
    {
        "type": "exact",
        "question": "What is the title of document SECURITY-002?",
        "expected": "Data Privacy Guidelines",
        "filename": "security_02.docx"
    },

    # 6 semantic questions
    {
        "type": "semantic",
        "question": "Which policy explains how workers should manage their time off?",
        "expected": "Annual Leave Policy",
        "filename": "hr_02.docx"
    },
    {
        "type": "semantic",
        "question": "Which document explains rules for protecting login credentials?",
        "expected": "Password Security Policy",
        "filename": "it_02.docx"
    },
    {
        "type": "semantic",
        "question": "Which policy covers employee presence and punctuality?",
        "expected": "Employee Attendance Policy",
        "filename": "hr_01.pdf"
    },
    {
        "type": "semantic",
        "question": "Which document describes repayment of employee travel expenses?",
        "expected": "Travel Reimbursement",
        "filename": "finance_02.docx"
    },
    {
        "type": "semantic",
        "question": "Which document describes controlling and recording people entering the workplace?",
        "expected": "Visitor Management",
        "filename": "operations_02.docx"
    },
    {
        "type": "semantic",
        "question": "Which policy explains protecting confidential personal information?",
        "expected": "Data Privacy Guidelines",
        "filename": "security_02.docx"
    }
]
