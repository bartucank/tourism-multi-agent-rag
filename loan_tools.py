"""Deterministic finance tools for the tourism assistant."""

import math
import re

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool


def _number(value):
    value = value.replace(" ", "")
    if "." in value and "," in value:
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")
    elif "," in value:
        tail = value.rsplit(",", 1)[1]
        value = value.replace(",", "." if len(tail) != 3 else "")
    elif "." in value:
        tail = value.rsplit(".", 1)[1]
        value = value.replace(".", "" if len(tail) == 3 else ".")
    return float(value)


def explicit_loan_inputs(messages):
    latest_user_index = next(
        (i for i in range(len(messages) - 1, -1, -1)
         if isinstance(messages[i], HumanMessage)),
        None,
    )
    if latest_user_index is None:
        return {}
    start_index = latest_user_index
    last_assistant_index = next(
        (i for i in range(latest_user_index - 1, -1, -1)
         if isinstance(messages[i], AIMessage)),
        None,
    )
    if (
        last_assistant_index is not None
        and "hesaplama için lütfen" in str(messages[last_assistant_index].content).lower()
    ):
        previous_user_index = next(
            (i for i in range(last_assistant_index - 1, -1, -1)
             if isinstance(messages[i], HumanMessage)),
            None,
        )
        if previous_user_index is not None:
            start_index = previous_user_index
    text = "\n".join(
        str(message.content)
        for message in messages[start_index:latest_user_index + 1]
        if isinstance(message, HumanMessage) and isinstance(message.content, str)
    )
    number = r"\d[\d.,]*"
    values = {}

    amount_pattern = re.compile(
        rf"(?<![\w.,])(?P<amount>{number})(?:\s*(?P<scale>bin|milyon|milyar))?"
        r"\s*(?:TL|TRY|lira|₺)\b|"
        rf"₺\s*(?P<prefixed>{number})",
        re.I,
    )
    amount_matches = list(amount_pattern.finditer(text))
    if amount_matches:
        match = amount_matches[-1]
        raw = match.group("amount") or match.group("prefixed")
        scale = (match.groupdict().get("scale") or "").lower()
        multiplier = {"bin": 1_000, "milyon": 1_000_000, "milyar": 1_000_000_000}.get(scale, 1)
        values["principal"] = _number(raw) * multiplier

    rate_pattern = re.compile(
        rf"(?:%\s*(?P<prefix>{number})|(?P<suffix>{number})\s*%|"
        rf"\b(?:yüzde|yuzde)\s*(?P<word>{number})\s*(?:oran)?\b)",
        re.I,
    )
    rate_matches = list(rate_pattern.finditer(text))
    if rate_matches:
        match = rate_matches[-1]
        raw = match.group("prefix") or match.group("suffix") or match.group("word")
        values["annual_interest_rate_percent"] = _number(raw)

    term_matches = list(
        re.finditer(r"\b(\d{1,3})\s*(?:ay|aylık|aylik|months?)\b", text, re.I)
    )
    if term_matches:
        values["term_months"] = int(term_matches[-1].group(1))

    return values


def loan_calculation_requested(messages):
    latest_user = next(
        (message for message in reversed(messages) if isinstance(message, HumanMessage)),
        None,
    )
    if latest_user is None or not isinstance(latest_user.content, str):
        return False
    return bool(re.search(
        r"\b(taksit\w*|geri ödeme\w*|geri odeme\w*|ödeme planı\w*|"
        r"odeme plani\w*|kredi hesapla\w*|payment\w*|installment\w*)\b",
        latest_user.content,
        re.I,
    ))


@tool
def calculate_loan_payment(
    principal: float,
    annual_interest_rate_percent: float,
    term_months: int,
) -> dict[str, float | int | str]:
    """Calculate equal monthly payments for a fixed-rate loan in TRY.

    Args:
        principal: Loan principal in Turkish lira.
        annual_interest_rate_percent: Fixed nominal annual interest rate as a percentage.
        term_months: Number of monthly payments.
    """
    values = (principal, annual_interest_rate_percent)
    if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
        raise ValueError("Principal and interest rate must be finite numbers")
    if principal <= 0 or principal > 1_000_000_000_000:
        raise ValueError("Principal must be between 0 and 1,000,000,000,000 TRY")
    if annual_interest_rate_percent < 0 or annual_interest_rate_percent > 1000:
        raise ValueError("Annual interest rate must be between 0% and 1000%")
    if isinstance(term_months, bool) or not isinstance(term_months, int):
        raise ValueError("Term must be a whole number of months")
    if not 1 <= term_months <= 600:
        raise ValueError("Term must be between 1 and 600 months")

    monthly_rate = annual_interest_rate_percent / 100 / 12
    if monthly_rate == 0:
        monthly_payment = principal / term_months
    else:
        monthly_payment = (
            principal * monthly_rate
            / (1 - (1 + monthly_rate) ** -term_months)
        )
    total_repayment = monthly_payment * term_months

    return {
        "principal_try": round(principal, 2),
        "annual_interest_rate_percent": round(annual_interest_rate_percent, 6),
        "term_months": term_months,
        "monthly_payment_try": round(monthly_payment, 2),
        "total_repayment_try": round(total_repayment, 2),
        "total_interest_try": round(total_repayment - principal, 2),
        "assumptions": (
            "Fixed nominal annual rate divided by 12; equal end-of-month "
            "payments; excludes fees, taxes, insurance, and variable-rate changes."
        ),
    }
