"""
Gemini-powered accounting agent.

Uses Gemini 2.5 Pro with function-calling to:
1. Interpret the multilingual task prompt
2. Plan the minimum set of API calls
3. Execute them against the Tripletex API
4. Self-correct on errors

All field names and schemas are verified against the official Tripletex OpenAPI spec.
"""

import asyncio
import base64
import json
import logging
import os
import time
from datetime import date
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed (e.g. in Docker), env vars come from Cloud Run

from google import genai
from google.genai import types

from tripletex_client import TripletexClient

logger = logging.getLogger(__name__)

# ── Tripletex tool definitions for Gemini function-calling ───────────
# Every tool below uses ONLY field names verified from the OpenAPI spec
# and live sandbox testing.

TRIPLETEX_TOOLS = [
    types.Tool(function_declarations=[
        # ── Employee ─────────────────────────────────────────────
        types.FunctionDeclaration(
            name="list_employees",
            description="List/search employees. Returns {fullResultSize, values: [...]}.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "firstName": types.Schema(type="STRING", description="Filter by first name"),
                    "lastName": types.Schema(type="STRING", description="Filter by last name"),
                    "email": types.Schema(type="STRING", description="Filter by email"),
                    "fields": types.Schema(type="STRING", description="Comma-separated fields, e.g. 'id,firstName,lastName,email'. Use '*' for all."),
                    "count": types.Schema(type="INTEGER", description="Max results"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="get_employee",
            description="Get a single employee by ID.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "id": types.Schema(type="INTEGER", description="Employee ID"),
                    "fields": types.Schema(type="STRING", description="Fields to return"),
                },
                required=["id"],
            ),
        ),
        types.FunctionDeclaration(
            name="create_employee",
            description=(
                "Create a new employee. "
                "Required fields: firstName, lastName, userType, department. "
                "Valid userType values: STANDARD (limited access), EXTENDED (full entitlements possible), NO_ACCESS (no login). "
                "department must be a JSON object string with id, e.g. {\"id\": 851682}. "
                "To get the department id, first call list_departments. "
                "To make someone an administrator (kontoadministrator), create with userType=EXTENDED then use grant_entitlement."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "firstName": types.Schema(type="STRING"),
                    "lastName": types.Schema(type="STRING"),
                    "email": types.Schema(type="STRING"),
                    "userType": types.Schema(type="STRING", description="STANDARD, EXTENDED, or NO_ACCESS"),
                    "department": types.Schema(type="STRING", description="JSON object string: {\"id\": <dept_id>}"),
                    "phoneNumberMobile": types.Schema(type="STRING"),
                    "phoneNumberHome": types.Schema(type="STRING"),
                    "phoneNumberWork": types.Schema(type="STRING"),
                    "dateOfBirth": types.Schema(type="STRING", description="YYYY-MM-DD"),
                    "nationalIdentityNumber": types.Schema(type="STRING"),
                    "address": types.Schema(type="STRING", description="JSON object string with addressLine1, postalCode, city"),
                    "allowInformationRegistration": types.Schema(type="BOOLEAN"),
                    "employeeNumber": types.Schema(type="STRING"),
                    "comments": types.Schema(type="STRING"),
                    "bankAccountNumber": types.Schema(type="STRING"),
                    "iban": types.Schema(type="STRING"),
                    "bic": types.Schema(type="STRING"),
                    "isContact": types.Schema(type="BOOLEAN"),
                },
                required=["firstName", "lastName", "userType", "department"],
            ),
        ),
        types.FunctionDeclaration(
            name="update_employee",
            description="Update an existing employee. You MUST first GET the employee with fields=* to get all current fields including 'version', then send the full body with modifications.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "id": types.Schema(type="INTEGER", description="Employee ID"),
                    "body": types.Schema(type="STRING", description="Full JSON body as string. Must include id, version, firstName, lastName, userType, department at minimum."),
                },
                required=["id", "body"],
            ),
        ),
        # ── Employee Entitlement (admin roles) ───────────────────
        types.FunctionDeclaration(
            name="grant_entitlement",
            description=(
                "Grant an entitlement/role to an employee. "
                "To make someone administrator (kontoadministrator): entitlementId=1 (ROLE_ADMINISTRATOR). "
                "The employee MUST have userType=EXTENDED for admin entitlements. "
                "customerId is the companyId from the employee object."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "employeeId": types.Schema(type="INTEGER", description="Employee ID"),
                    "entitlementId": types.Schema(type="INTEGER", description="1=ROLE_ADMINISTRATOR, and others"),
                    "customerId": types.Schema(type="INTEGER", description="Company ID (from employee.companyId)"),
                },
                required=["employeeId", "entitlementId", "customerId"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_entitlements",
            description="List entitlements for an employee.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "employeeId": types.Schema(type="INTEGER", description="Employee ID"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
                required=["employeeId"],
            ),
        ),
        # ── Employment ───────────────────────────────────────────
        types.FunctionDeclaration(
            name="create_employment",
            description=(
                "Create an employment relationship for an employee. "
                "POST /employee/employment. "
                "Required: employee:{id}, startDate. "
                "Optional: endDate, isMainEmployer (default true). "
                "Do NOT include occupationCode here — it goes on employment details. "
                "After creating employment, use create_employment_details to set salary, occupation code, etc."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "body": types.Schema(type="STRING", description="Full JSON body as string"),
                },
                required=["body"],
            ),
        ),
        types.FunctionDeclaration(
            name="create_employment_details",
            description=(
                "Create employment details (salary, work percentage, occupation) for an employment. "
                "POST /employee/employment/details. "
                "Required: employment:{id}, date (YYYY-MM-DD). "
                "Optional: percentageOfFullTimeEquivalent (e.g. 100.0), "
                "annualSalary (yearly salary amount), "
                "occupationCode:{id} (get id from list_occupation_codes), "
                "remunerationType (MONTHLY_WAGE or HOURLY_WAGE), "
                "employmentType (ORDINARY, MARITIME, FREELANCE, etc.). "
                "Do NOT use monthlyWage or paymentType — these fields do not exist. "
                "The employment ID comes from create_employment response."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "body": types.Schema(type="STRING", description="Full JSON body as string"),
                },
                required=["body"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_occupation_codes",
            description=(
                "List/search occupation codes (yrkeskoder/STYRK codes). "
                "GET /employee/employment/occupationCode. "
                "Use nameNO or code parameter to search. Returns id, code, nameNO."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "nameNO": types.Schema(type="STRING", description="Search by Norwegian name"),
                    "code": types.Schema(type="STRING", description="Search by STYRK code"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        # ── Customer ─────────────────────────────────────────────
        types.FunctionDeclaration(
            name="list_customers",
            description="List/search customers. Can filter by name, email, or organizationNumber.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "name": types.Schema(type="STRING", description="Filter by name"),
                    "email": types.Schema(type="STRING"),
                    "organizationNumber": types.Schema(type="STRING"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="create_customer",
            description=(
                "Create a new customer. Required: name. "
                "Always set isCustomer=true for customers. "
                "Verified fields: name, email, phoneNumber, phoneNumberMobile, organizationNumber, "
                "isCustomer, isSupplier, isPrivateIndividual, language, invoiceSendMethod, "
                "invoicesDueIn, invoicesDueInType, postalAddress, physicalAddress, accountManager, department."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "name": types.Schema(type="STRING", description="Customer name"),
                    "email": types.Schema(type="STRING"),
                    "phoneNumber": types.Schema(type="STRING"),
                    "phoneNumberMobile": types.Schema(type="STRING"),
                    "organizationNumber": types.Schema(type="STRING"),
                    "postalAddress": types.Schema(type="STRING", description="JSON object string with addressLine1, postalCode, city"),
                    "physicalAddress": types.Schema(type="STRING", description="JSON object string"),
                    "isCustomer": types.Schema(type="BOOLEAN"),
                    "isSupplier": types.Schema(type="BOOLEAN"),
                    "isPrivateIndividual": types.Schema(type="BOOLEAN"),
                    "language": types.Schema(type="STRING", description="e.g. NO, EN"),
                    "invoiceSendMethod": types.Schema(type="STRING", description="EMAIL, EHF, EFAKTURA, VIPPS, PAPER, MANUAL"),
                    "invoicesDueIn": types.Schema(type="INTEGER", description="Payment terms in days"),
                    "invoicesDueInType": types.Schema(type="STRING", description="DAYS, MONTHS, RECURRING_DAY_OF_MONTH"),
                    "accountManager": types.Schema(type="STRING", description="JSON object string: {\"id\": <employee_id>}"),
                    "department": types.Schema(type="STRING", description="JSON object string: {\"id\": <dept_id>}"),
                },
                required=["name"],
            ),
        ),
        types.FunctionDeclaration(
            name="update_customer",
            description="Update an existing customer. First GET with fields=* to get version.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "id": types.Schema(type="INTEGER"),
                    "body": types.Schema(type="STRING", description="Full JSON body as string"),
                },
                required=["id", "body"],
            ),
        ),
        # ── Product ──────────────────────────────────────────────
        types.FunctionDeclaration(
            name="list_products",
            description="List/search products.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "name": types.Schema(type="STRING"),
                    "number": types.Schema(type="STRING"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="create_product",
            description=(
                "Create a new product. Required: name. "
                "Price fields: priceExcludingVatCurrency, priceIncludingVatCurrency. "
                "vatType is a JSON object string with id. IMPORTANT: valid vatType IDs are company-specific. "
                "Do NOT assume any particular ID. If the task requires a VAT rate, first try without vatType "
                "(the system assigns a default). Only specify vatType if a specific rate is explicitly required "
                "AND you know the correct ID from list_vat_types. "
                "Other fields: number, description, costExcludingVatCurrency, productUnit, account, department."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "name": types.Schema(type="STRING"),
                    "number": types.Schema(type="STRING", description="Product number"),
                    "description": types.Schema(type="STRING"),
                    "priceExcludingVatCurrency": types.Schema(type="NUMBER"),
                    "priceIncludingVatCurrency": types.Schema(type="NUMBER"),
                    "costExcludingVatCurrency": types.Schema(type="NUMBER"),
                    "vatType": types.Schema(type="STRING", description="JSON: {\"id\":3} for 25% VAT, {\"id\":6} for 0%"),
                    "productUnit": types.Schema(type="STRING", description="JSON: {\"id\": <unit_id>}"),
                    "account": types.Schema(type="STRING", description="JSON: {\"id\": <account_id>}"),
                    "department": types.Schema(type="STRING", description="JSON: {\"id\": <dept_id>}"),
                },
                required=["name"],
            ),
        ),
        # ── Order ────────────────────────────────────────────────
        types.FunctionDeclaration(
            name="create_order",
            description=(
                "Create a new order. "
                "Body must include: customer:{id}, orderDate (YYYY-MM-DD), deliveryDate (YYYY-MM-DD). "
                "orderLines is an array of {product:{id}, count, unitPriceExcludingVatCurrency (or unitCostPrice)}."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "body": types.Schema(type="STRING", description="Full JSON body as string"),
                },
                required=["body"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_orders",
            description="List/search orders. IMPORTANT: orderDateFrom and orderDateTo are REQUIRED by the API.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "orderDateFrom": types.Schema(type="STRING", description="YYYY-MM-DD, REQUIRED"),
                    "orderDateTo": types.Schema(type="STRING", description="YYYY-MM-DD, REQUIRED"),
                    "customerId": types.Schema(type="STRING"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
                required=["orderDateFrom", "orderDateTo"],
            ),
        ),
        # ── Invoice ──────────────────────────────────────────────
        types.FunctionDeclaration(
            name="create_invoice",
            description=(
                "Create an invoice from an order. "
                "Body must include: invoiceDate, invoiceDueDate, orders:[{id}]. "
                "Note: you must create an order first, then reference it."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "body": types.Schema(type="STRING", description="Full JSON body as string"),
                },
                required=["body"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_invoices",
            description=(
                "List/search invoices. IMPORTANT: invoiceDateFrom and invoiceDateTo are REQUIRED. "
                "Valid fields: id, invoiceNumber, invoiceDate, invoiceDueDate, customer, amount, amountCurrency, "
                "amountExcludingVat, amountOutstanding, isCreditNote, orders, orderLines, voucher, isApproved. "
                "Do NOT use 'number' or 'order' as fields — they don't exist on InvoiceDTO."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "invoiceDateFrom": types.Schema(type="STRING", description="YYYY-MM-DD, REQUIRED"),
                    "invoiceDateTo": types.Schema(type="STRING", description="YYYY-MM-DD, REQUIRED"),
                    "customerId": types.Schema(type="STRING"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
                required=["invoiceDateFrom", "invoiceDateTo"],
            ),
        ),
        types.FunctionDeclaration(
            name="create_credit_note",
            description=(
                "Create a credit note for an existing invoice. Uses PUT /invoice/{id}/:createCreditNote. "
                "This reverses/cancels the invoice. Parameters are QUERY PARAMS (not body). "
                "Required: invoiceId, date. Optional: comment, creditNoteEmail, sendToCustomer."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "invoiceId": types.Schema(type="INTEGER", description="ID of the invoice to credit"),
                    "date": types.Schema(type="STRING", description="Credit note date YYYY-MM-DD"),
                    "comment": types.Schema(type="STRING"),
                    "creditNoteEmail": types.Schema(type="STRING"),
                    "sendToCustomer": types.Schema(type="BOOLEAN"),
                },
                required=["invoiceId", "date"],
            ),
        ),
        # ── Payment ──────────────────────────────────────────────
        types.FunctionDeclaration(
            name="create_payment",
            description=(
                "Register a payment on an invoice. Uses PUT /invoice/{id}/:payment with QUERY PARAMETERS. "
                "paymentTypeId: MUST come from list_invoice_payment_types (NOT list_payment_types which is for outgoing). "
                "paidAmount: the payment amount in company currency (NOK)."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "invoiceId": types.Schema(type="INTEGER"),
                    "paymentDate": types.Schema(type="STRING", description="YYYY-MM-DD"),
                    "paymentTypeId": types.Schema(type="INTEGER"),
                    "paidAmount": types.Schema(type="NUMBER", description="Amount in company currency (NOK)"),
                },
                required=["invoiceId", "paymentDate", "paymentTypeId", "paidAmount"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_payment_types",
            description="List OUTGOING payment types (for paying bills). Endpoint: /ledger/paymentTypeOut. For INVOICE payments (receiving money), use list_invoice_payment_types instead.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "fields": types.Schema(type="STRING", description="Use 'id,description' — there is no 'name' field"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="list_invoice_payment_types",
            description="List INCOMING payment types for invoice payments (receiving money from customers). Endpoint: /invoice/paymentType. Use these IDs for create_payment.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "fields": types.Schema(type="STRING", description="Use 'id,description'"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        # ── Travel Expense ───────────────────────────────────────
        types.FunctionDeclaration(
            name="list_travel_expenses",
            description="List travel expense reports.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "employeeId": types.Schema(type="STRING"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="create_travel_expense",
            description=(
                "Create a travel expense report. Required: employee:{id}. "
                "IMPORTANT: To use per diem, you MUST include a 'travelDetails' object with "
                "departureDate, returnDate, departureTime, returnTime (HH:MM format), "
                "departureFrom, destination. This creates a 'reiseregning' (travel report). "
                "Without travelDetails, it creates an 'ansattutlegg' (employee disbursement) "
                "which does NOT support per diem. "
                "Optional: title, date (YYYY-MM-DD), project:{id}, department:{id}."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "body": types.Schema(type="STRING", description="Full JSON body as string"),
                },
                required=["body"],
            ),
        ),
        types.FunctionDeclaration(
            name="create_travel_expense_cost",
            description=(
                "Add a cost line to a travel expense report. Endpoint: POST /travelExpense/cost. "
                "Required: travelExpense:{id}, amountCurrencyIncVat, paymentType:{id}. "
                "Optional: date, category (string description), costCategory:{id}, comments, "
                "isPaidByEmployee (default true), isChargeable, currency:{id}, vatType:{id}. "
                "First get paymentType IDs from list_travel_payment_types."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "body": types.Schema(type="STRING", description="Full JSON body as string"),
                },
                required=["body"],
            ),
        ),
        types.FunctionDeclaration(
            name="create_per_diem_compensation",
            description=(
                "Add per diem compensation to a travel expense report. Endpoint: POST /travelExpense/perDiemCompensation. "
                "Required: travelExpense:{id}, location (string). "
                "Optional: count (number of days), rate, countryCode, overnightAccommodation (HOTEL/NONE), "
                "isDeductionForBreakfast, isDeductionForLunch, isDeductionForDinner, address."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "body": types.Schema(type="STRING", description="Full JSON body as string"),
                },
                required=["body"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_travel_payment_types",
            description="List travel expense payment types. Endpoint: /travelExpense/paymentType. Uses 'description' field.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="list_travel_cost_categories",
            description="List travel expense cost categories. Endpoint: /travelExpense/costCategory.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="delete_travel_expense",
            description="Delete a travel expense report by ID.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "id": types.Schema(type="INTEGER"),
                },
                required=["id"],
            ),
        ),
        # ── Timesheet ────────────────────────────────────────────
        types.FunctionDeclaration(
            name="create_timesheet_entry",
            description=(
                "Log hours on a project activity. Endpoint: POST /timesheet/entry. "
                "Required: employee:{id}, activity:{id}, date (YYYY-MM-DD), hours (number). "
                "Optional: project:{id}, comment, chargeable (bool). "
                "IMPORTANT: date MUST be >= the project's startDate. If you get a date validation "
                "error, read the project start date from the error and retry with a valid date "
                "(use the project start date or today, whichever is later). NEVER give up on timesheet logging."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "employeeId": types.Schema(type="INTEGER", description="Employee ID"),
                    "activityId": types.Schema(type="INTEGER", description="Activity ID (from list_activities)"),
                    "projectId": types.Schema(type="INTEGER", description="Project ID (optional)"),
                    "date": types.Schema(type="STRING", description="YYYY-MM-DD, must be >= project start date"),
                    "hours": types.Schema(type="NUMBER", description="Number of hours to log"),
                    "comment": types.Schema(type="STRING"),
                },
                required=["employeeId", "activityId", "date", "hours"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_activities",
            description="List activities available for timesheet entries. Use to find activity IDs like 'Design', 'Development', etc.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "name": types.Schema(type="STRING", description="Filter by activity name"),
                    "fields": types.Schema(type="STRING", description="e.g. 'id,name'"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        # ── Project ──────────────────────────────────────────────
        types.FunctionDeclaration(
            name="list_projects",
            description="List/search projects.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "name": types.Schema(type="STRING"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="create_project",
            description=(
                "Create a new project. "
                "Body must include: name, projectManager:{id}, startDate. "
                "Optional: number (string, auto-assigned if omitted), customer:{id}, endDate, isClosed, isInternal, "
                "isFixedPrice (bool), fixedprice (number, NOTE: lowercase 'p'!). "
                "Do NOT include 'number' unless the task specifies one — Tripletex auto-assigns it. "
                "Do NOT use 'fixedPrice' (capital P) — use 'fixedprice' (lowercase p)."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "body": types.Schema(type="STRING", description="Full JSON body as string"),
                },
                required=["body"],
            ),
        ),
        # ── Department ───────────────────────────────────────────
        types.FunctionDeclaration(
            name="list_departments",
            description="List departments. Returns id, name, departmentNumber, etc.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "name": types.Schema(type="STRING"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="create_department",
            description="Create a new department. Fields: name (required), departmentNumber, departmentManager:{id}.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "name": types.Schema(type="STRING"),
                    "departmentNumber": types.Schema(type="STRING"),
                    "departmentManager": types.Schema(type="STRING", description="JSON: {\"id\": <employee_id>}"),
                },
                required=["name"],
            ),
        ),
        # ── Ledger ───────────────────────────────────────────────
        types.FunctionDeclaration(
            name="list_accounts",
            description="List chart of accounts (ledger accounts).",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "number": types.Schema(type="STRING", description="Account number"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="list_vat_types",
            description="List VAT types. IDs are company-specific. Valid fields: id, name, percentage, number. Do NOT use 'rate' or 'type' — these don't exist on VatTypeDTO.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "fields": types.Schema(type="STRING", description="Valid: id,name,percentage,number. NOT rate or type."),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="list_vouchers",
            description="List vouchers. dateTo is EXCLUSIVE — to get vouchers for a single date, set dateTo = dateFrom + 1 day. Valid fields: id, date, number, year, description, postings.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "dateFrom": types.Schema(type="STRING", description="YYYY-MM-DD inclusive start"),
                    "dateTo": types.Schema(type="STRING", description="YYYY-MM-DD exclusive end"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="create_voucher",
            description=(
                "Create a voucher with postings. Body as full JSON string. "
                "IMPORTANT: Use 'amountGross' (NOT 'amount') for posting amounts — the API only uses gross amounts. "
                "Postings must balance (sum of amountGross = 0). Example: "
                "{date:'2025-03-01', description:'Expense', postings:[{account:{id:X}, amountGross:1000}, {account:{id:Y}, amountGross:-1000}]}"
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "body": types.Schema(type="STRING"),
                },
                required=["body"],
            ),
        ),
        types.FunctionDeclaration(
            name="delete_voucher",
            description="Delete a voucher by ID.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "id": types.Schema(type="INTEGER"),
                },
                required=["id"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_postings",
            description="List ledger postings. Endpoint: /ledger/posting. dateTo is EXCLUSIVE — to get postings for a single date, set dateTo = dateFrom + 1 day.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "dateFrom": types.Schema(type="STRING", description="YYYY-MM-DD inclusive start"),
                    "dateTo": types.Schema(type="STRING", description="YYYY-MM-DD exclusive end"),
                    "accountId": types.Schema(type="STRING"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        # ── Contact ──────────────────────────────────────────────
        types.FunctionDeclaration(
            name="list_contacts",
            description="List contacts for a customer. Endpoint: /contact",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "customerId": types.Schema(type="INTEGER"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="create_contact",
            description="Create a contact for a customer. Body as full JSON string.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "body": types.Schema(type="STRING"),
                },
                required=["body"],
            ),
        ),
        # ── Salary / Payroll ─────────────────────────────────────
        types.FunctionDeclaration(
            name="list_salary_types",
            description="List salary types (lønnarter). Returns id, name, number, description.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "name": types.Schema(type="STRING"),
                    "number": types.Schema(type="STRING"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="create_salary_transaction",
            description=(
                "Create a salary/payroll transaction (lønnskjøring). "
                "Required: year, month, payslips array. "
                "Each payslip: {employee:{id}, date:'YYYY-MM-DD', year, month, "
                "specifications:[{salaryType:{id}, rate:<amount>, count:1, description:'...'}]}. "
                "Use list_salary_types to find correct salaryType IDs. "
                "For base salary, find the 'Fastlønn' or 'Månedslønn' type. "
                "For bonus, find 'Bonus' or 'Tillegg' type."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "body": types.Schema(type="STRING", description="Full JSON body as string"),
                    "generateTaxDeduction": types.Schema(type="BOOLEAN", description="Whether to auto-generate tax deduction"),
                },
                required=["body"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_salary_payslips",
            description="List salary payslips.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "employeeId": types.Schema(type="STRING"),
                    "yearFrom": types.Schema(type="INTEGER"),
                    "yearTo": types.Schema(type="INTEGER"),
                    "monthFrom": types.Schema(type="INTEGER"),
                    "monthTo": types.Schema(type="INTEGER"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        # ── Supplier ─────────────────────────────────────────────
        types.FunctionDeclaration(
            name="list_supplier_invoices",
            description=(
                "List/search supplier invoices (leverandørfakturaer). "
                "GET /supplierInvoice. "
                "Use supplierId, invoiceDateFrom/invoiceDateTo to filter. "
                "NOTE: 'isClosed' and 'amountOutstanding' are NOT valid fields on SupplierInvoiceDTO — do NOT use them. "
                "Valid fields: id, invoiceNumber, invoiceDate, supplier, amount, amountCurrency, kid, paymentTypeId, dueDate."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "supplierId": types.Schema(type="INTEGER", description="Filter by supplier ID"),
                    "invoiceDateFrom": types.Schema(type="STRING", description="YYYY-MM-DD"),
                    "invoiceDateTo": types.Schema(type="STRING", description="YYYY-MM-DD"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="pay_supplier_invoice",
            description=(
                "Register payment on a supplier invoice. "
                "PUT /supplierInvoice/{id}/:registerPayment. "
                "Required: invoiceId, paymentDate, paymentTypeId, paidAmount. "
                "Use list_payment_types to find outgoing payment type IDs."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "invoiceId": types.Schema(type="INTEGER", description="Supplier invoice ID"),
                    "paymentDate": types.Schema(type="STRING", description="YYYY-MM-DD"),
                    "paymentTypeId": types.Schema(type="INTEGER", description="Payment type ID from list_payment_types"),
                    "paidAmount": types.Schema(type="NUMBER", description="Amount paid"),
                },
                required=["invoiceId", "paymentDate", "paymentTypeId", "paidAmount"],
            ),
        ),
        types.FunctionDeclaration(
            name="create_supplier_invoice",
            description=(
                "Create a supplier invoice (leverandørfaktura). Endpoint: POST /supplierInvoice. "
                "Required: invoiceDate, supplier:{id}. "
                "The invoice MUST include a voucher with postings for the accounting entries. "
                "Schema: {invoiceNumber, invoiceDate, invoiceDueDate, supplier:{id}, "
                "voucher:{date, description, postings:[{account:{id}, amountGross, vatType:{id}, description}]}}"
                "CRITICAL: Use 'amountGross' NOT 'amount' — the API only uses gross amounts. "
                "CRITICAL: You MUST include BOTH debit AND credit postings. "
                "Example: supplier invoice for 50050 NOK incl 25% VAT on account 6300: "
                "voucher.postings = [{account:{id:acct6300}, amountGross:40040, vatType:{id:vat25_input}}, "
                "{account:{id:acct2400_supplier}, amountGross:-50050}]"
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "body": types.Schema(type="STRING", description="Full JSON body as string"),
                },
                required=["body"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_suppliers",
            description="List/search suppliers. Endpoint: GET /supplier.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "name": types.Schema(type="STRING", description="Filter by supplier name"),
                    "organizationNumber": types.Schema(type="STRING"),
                    "fields": types.Schema(type="STRING"),
                    "count": types.Schema(type="INTEGER"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="create_supplier",
            description="Create a new supplier. Required: name. Optional: organizationNumber, email, phoneNumber, isSupplier (default true), isCustomer.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "name": types.Schema(type="STRING"),
                    "organizationNumber": types.Schema(type="STRING"),
                    "email": types.Schema(type="STRING"),
                    "phoneNumber": types.Schema(type="STRING"),
                    "isCustomer": types.Schema(type="BOOLEAN"),
                },
                required=["name"],
            ),
        ),
        # ── Generic / fallback ───────────────────────────────────
        types.FunctionDeclaration(
            name="tripletex_api_call",
            description=(
                "Make an arbitrary Tripletex API call for endpoints not covered by other tools. "
                "Examples: /invoice/{id}/:createCreditNote, /currency, /company, "
                "/invoice/paymentType, /travelExpense/cost, /travelExpense/perDiemCompensation, "
                "/order/orderline, /supplier, /contact, /activity, etc. "
                "Do NOT use for /authentications, /accessrights, or /users — these will 404."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "method": types.Schema(type="STRING", description="HTTP method: GET, POST, PUT, DELETE"),
                    "path": types.Schema(type="STRING", description="API path, e.g. '/invoice/123/:createCreditNote'"),
                    "params": types.Schema(type="STRING", description="Query parameters as JSON object string"),
                    "body": types.Schema(type="STRING", description="Request body as JSON string"),
                },
                required=["method", "path"],
            ),
        ),
    ])
]


# ── Tool execution (maps function calls → Tripletex API) ────────────

def execute_tool(client: TripletexClient, name: str, args: dict) -> dict:
    """Execute a single Gemini function-call against the Tripletex API."""

    def _fields_params(args: dict) -> dict:
        p = {}
        if "fields" in args:
            p["fields"] = args["fields"]
        if "count" in args:
            p["count"] = args["count"]
        return p

    def _safe_json(s: str) -> Any:
        if isinstance(s, (dict, list)):
            return s
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            # Fix double-braces from parallel tool calls (e.g. "{{"id": 123}}")
            if isinstance(s, str) and "{{" in s:
                fixed = s.replace("{{", "{").replace("}}", "}")
                try:
                    return json.loads(fixed)
                except (json.JSONDecodeError, TypeError):
                    pass
            return s

    match name:
        # ── Employee ─────────────────────────────────────────
        case "list_employees":
            params = _fields_params(args)
            for k in ("firstName", "lastName", "email"):
                if k in args:
                    params[k] = args[k]
            return client.get("/employee", params=params)

        case "get_employee":
            return client.get(f"/employee/{args['id']}", params={"fields": args.get("fields", "*")})

        case "create_employee":
            body = {}
            # Direct string/bool fields
            for k in ("firstName", "lastName", "email", "phoneNumberMobile",
                       "phoneNumberHome", "phoneNumberWork", "dateOfBirth",
                       "nationalIdentityNumber", "userType", "employeeNumber",
                       "comments", "bankAccountNumber", "iban", "bic"):
                if k in args and args[k] is not None:
                    body[k] = args[k]
            # Boolean fields
            for k in ("allowInformationRegistration", "isContact"):
                if k in args and args[k] is not None:
                    body[k] = args[k]
            # JSON object fields
            for k in ("department", "address"):
                if k in args and args[k] is not None:
                    body[k] = _safe_json(args[k])
            return client.create("/employee", body)

        case "update_employee":
            body = _safe_json(args["body"])
            return client.update(f"/employee/{args['id']}", body)

        # ── Entitlement ──────────────────────────────────────
        case "grant_entitlement":
            body = {
                "employee": {"id": args["employeeId"]},
                "entitlementId": args["entitlementId"],
                "customer": {"id": args["customerId"]},
            }
            return client.create("/employee/entitlement", body)

        case "list_entitlements":
            params = _fields_params(args)
            params["employeeId"] = args["employeeId"]
            return client.get("/employee/entitlement", params=params)

        # ── Employment ────────────────────────────────────────
        case "create_employment":
            body = _safe_json(args["body"])
            return client.create("/employee/employment", body)

        case "create_employment_details":
            body = _safe_json(args["body"])
            return client.create("/employee/employment/details", body)

        case "list_occupation_codes":
            params = _fields_params(args)
            for k in ("nameNO", "code"):
                if k in args:
                    params[k] = args[k]
            return client.get("/employee/employment/occupationCode", params=params)

        # ── Customer ─────────────────────────────────────────
        case "list_customers":
            params = _fields_params(args)
            for k in ("name", "email", "organizationNumber"):
                if k in args:
                    params[k] = args[k]
            return client.get("/customer", params=params)

        case "create_customer":
            body = {}
            for k in ("name", "email", "phoneNumber", "phoneNumberMobile",
                       "organizationNumber", "language", "invoiceSendMethod",
                       "invoicesDueInType"):
                if k in args and args[k] is not None:
                    body[k] = args[k]
            for k in ("isCustomer", "isSupplier", "isPrivateIndividual"):
                if k in args and args[k] is not None:
                    body[k] = args[k]
            if "invoicesDueIn" in args:
                body["invoicesDueIn"] = args["invoicesDueIn"]
            for nested in ("postalAddress", "physicalAddress", "accountManager", "department"):
                if nested in args and args[nested] is not None:
                    body[nested] = _safe_json(args[nested])
            if "isCustomer" not in body:
                body["isCustomer"] = True
            return client.create("/customer", body)

        case "update_customer":
            body = _safe_json(args["body"])
            return client.update(f"/customer/{args['id']}", body)

        # ── Product ──────────────────────────────────────────
        case "list_products":
            params = _fields_params(args)
            for k in ("name", "number"):
                if k in args:
                    params[k] = args[k]
            return client.get("/product", params=params)

        case "create_product":
            body = {}
            for k in ("name", "number", "description"):
                if k in args and args[k] is not None:
                    body[k] = args[k]
            for k in ("priceExcludingVatCurrency", "priceIncludingVatCurrency", "costExcludingVatCurrency"):
                if k in args and args[k] is not None:
                    body[k] = args[k]
            for nested in ("vatType", "productUnit", "account", "department"):
                if nested in args and args[nested] is not None:
                    body[nested] = _safe_json(args[nested])
            return client.create("/product", body)

        # ── Order ────────────────────────────────────────────
        case "list_orders":
            params = _fields_params(args)
            for k in ("customerId", "orderDateFrom", "orderDateTo"):
                if k in args:
                    params[k] = args[k]
            return client.get("/order", params=params)

        case "create_order":
            body = _safe_json(args["body"])
            return client.create("/order", body)

        # ── Invoice ──────────────────────────────────────────
        case "list_invoices":
            params = _fields_params(args)
            for k in ("customerId", "invoiceDateFrom", "invoiceDateTo"):
                if k in args:
                    params[k] = args[k]
            return client.get("/invoice", params=params)

        case "create_invoice":
            body = _safe_json(args["body"])
            result = client.create("/invoice", body)
            if "_error" in result:
                detail = str(result.get("detail", "")).lower()
                if "bankkontonummer" in detail:
                    # Set bank account number on account 1920, then retry once.
                    acct = client.get(
                        "/ledger/account",
                        params={"number": "1920", "fields": "id,version,name,number"},
                    )
                    if "_error" not in acct:
                        values = acct.get("values", []) if isinstance(acct, dict) else []
                        if values:
                            acc = values[0]
                            acc_id = acc.get("id")
                            acc_version = acc.get("version")
                            if acc_id is not None and acc_version is None:
                                acc_full = client.get(
                                    f"/ledger/account/{acc_id}",
                                    params={"fields": "*"},
                                )
                                if "_error" not in acc_full:
                                    acc_val = acc_full.get("value", acc_full)
                                    if isinstance(acc_val, dict):
                                        acc_version = acc_val.get("version")
                            if acc_id is not None and acc_version is not None:
                                bank_body = {
                                    "id": acc_id,
                                    "version": acc_version,
                                    "name": acc.get("name", "Bankinnskudd"),
                                    "number": acc.get("number", 1920),
                                    "bankAccountNumber": "12345678903",
                                    "isBankAccount": True,
                                }
                                client.put(f"/ledger/account/{acc_id}", json=bank_body)
                                return client.create("/invoice", body)
            return result

        case "create_credit_note":
            invoice_id = args["invoiceId"]
            params = {"date": args["date"]}
            if "comment" in args:
                params["comment"] = args["comment"]
            if "creditNoteEmail" in args:
                params["creditNoteEmail"] = args["creditNoteEmail"]
            if "sendToCustomer" in args:
                params["sendToCustomer"] = args["sendToCustomer"]
            result = client.put(f"/invoice/{invoice_id}/:createCreditNote", params=params)
            if "_error" in result:
                detail = str(result.get("detail", "")).lower()
                # If date is before invoice date, retry with the invoice date.
                if "fakturadato" in detail and "dato" in detail and "ikke" in detail:
                    inv = client.get(f"/invoice/{invoice_id}", params={"fields": "invoiceDate"})
                    if "_error" not in inv:
                        inv_val = inv.get("value", inv)
                        inv_date = inv_val.get("invoiceDate") if isinstance(inv_val, dict) else None
                        if inv_date:
                            params["date"] = inv_date
                            return client.put(f"/invoice/{invoice_id}/:createCreditNote", params=params)
            return result

        # ── Payment ──────────────────────────────────────────
        case "create_payment":
            invoice_id = args["invoiceId"]
            return client.put(
                f"/invoice/{invoice_id}/:payment",
                params={
                    "paymentDate": args["paymentDate"],
                    "paymentTypeId": args["paymentTypeId"],
                    "paidAmount": args["paidAmount"],
                },
            )

        case "list_payment_types":
            params = _fields_params(args)
            if params.get("fields") and "name" in params["fields"]:
                params["fields"] = params["fields"].replace("name", "description")
            return client.get("/ledger/paymentTypeOut", params=params)

        case "list_invoice_payment_types":
            params = _fields_params(args)
            if params.get("fields") and "name" in params["fields"]:
                params["fields"] = params["fields"].replace("name", "description")
            return client.get("/invoice/paymentType", params=params)

        # ── Timesheet ────────────────────────────────────────
        case "create_timesheet_entry":
            body = {
                "employee": {"id": args["employeeId"]},
                "activity": {"id": args["activityId"]},
                "date": args["date"],
                "hours": args["hours"],
            }
            if "projectId" in args and args["projectId"]:
                body["project"] = {"id": args["projectId"]}
            if "comment" in args:
                body["comment"] = args["comment"]
            return client.create("/timesheet/entry", body)

        case "list_activities":
            params = _fields_params(args)
            if "name" in args:
                params["name"] = args["name"]
            return client.get("/activity", params=params)

        # ── Travel Expense ───────────────────────────────────
        case "list_travel_expenses":
            params = _fields_params(args)
            if "employeeId" in args:
                params["employeeId"] = args["employeeId"]
            return client.get("/travelExpense", params=params)

        case "create_travel_expense":
            body = _safe_json(args["body"])
            return client.create("/travelExpense", body)

        case "create_travel_expense_cost":
            body = _safe_json(args["body"])
            return client.create("/travelExpense/cost", body)

        case "create_per_diem_compensation":
            body = _safe_json(args["body"])
            return client.create("/travelExpense/perDiemCompensation", body)

        case "list_travel_payment_types":
            params = _fields_params(args)
            return client.get("/travelExpense/paymentType", params=params)

        case "list_travel_cost_categories":
            params = _fields_params(args)
            return client.get("/travelExpense/costCategory", params=params)

        case "delete_travel_expense":
            return client.delete(f"/travelExpense/{args['id']}")

        # ── Project ──────────────────────────────────────────
        case "list_projects":
            params = _fields_params(args)
            if "name" in args:
                params["name"] = args["name"]
            if "fields" in params:
                fields = [f.strip() for f in str(params["fields"]).split(",") if f.strip()]
                fields = [f for f in fields if f.lower() != "fixedprice"]
                if fields:
                    params["fields"] = ",".join(fields)
                else:
                    params.pop("fields", None)
            return client.get("/project", params=params)

        case "create_project":
            body = _safe_json(args["body"])
            return client.create("/project", body)

        # ── Department ───────────────────────────────────────
        case "list_departments":
            params = _fields_params(args)
            if "name" in args:
                params["name"] = args["name"]
            return client.get("/department", params=params)

        case "create_department":
            body = {"name": args["name"]}
            if "departmentNumber" in args:
                body["departmentNumber"] = args["departmentNumber"]
            if "departmentManager" in args:
                body["departmentManager"] = _safe_json(args["departmentManager"])
            return client.create("/department", body)

        # ── Ledger ───────────────────────────────────────────
        case "list_accounts":
            params = _fields_params(args)
            if "number" in args:
                params["number"] = args["number"]
            return client.get("/ledger/account", params=params)

        case "list_vat_types":
            params = _fields_params(args)
            return client.get("/ledger/vatType", params=params)

        case "list_vouchers":
            params = _fields_params(args)
            for k in ("dateFrom", "dateTo"):
                if k in args:
                    params[k] = args[k]
            # Fix: dateTo is EXCLUSIVE in Tripletex. If dateFrom == dateTo, bump dateTo by 1 day.
            if "dateFrom" in params and "dateTo" in params and params["dateFrom"] == params["dateTo"]:
                from datetime import timedelta
                dt = date.fromisoformat(params["dateTo"])
                params["dateTo"] = (dt + timedelta(days=1)).isoformat()
            return client.get("/ledger/voucher", params=params)

        case "create_voucher":
            body = _safe_json(args["body"])
            # CRITICAL: Fix postings for the Tripletex API.
            # 1. Row 0 is reserved for system-generated postings → always set row=1,2,...
            # 2. API only uses GROSS amounts: "amount" is ignored, must use "amountGross"
            if "postings" in body and isinstance(body["postings"], list):
                for i, posting in enumerate(body["postings"]):
                    if isinstance(posting, dict):
                        posting["row"] = i + 1  # Always override row
                        if "amountGross" not in posting and "amount" in posting:
                            posting["amountGross"] = posting.pop("amount")
                        if "amountGross" in posting and "amountGrossCurrency" not in posting:
                            posting["amountGrossCurrency"] = posting["amountGross"]
                # Pre-flight balance check: only check postings WITHOUT vatType (VAT adds system amounts)
                has_vat = any(isinstance(p, dict) and p.get("vatType") and p.get("vatType", {}).get("id", 0) != 0 for p in body["postings"])
                if not has_vat:
                    total = sum(
                        p.get("amountGross", 0) for p in body["postings"]
                        if isinstance(p, dict)
                    )
                    if abs(total) > 0.01:
                        return {
                            "_error": True,
                            "detail": f"BALANCE ERROR (caught before API call): Postings sum to {total}, but must be 0. "
                                       f"Fix the amountGross values so debits (positive) and credits (negative) balance exactly."
                        }
            return client.create("/ledger/voucher", body)

        case "delete_voucher":
            return client.delete(f"/ledger/voucher/{args['id']}")

        case "list_postings":
            params = _fields_params(args)
            for k in ("dateFrom", "dateTo", "accountId"):
                if k in args:
                    params[k] = args[k]
            # Fix: dateTo is EXCLUSIVE in Tripletex. If dateFrom == dateTo, bump dateTo by 1 day.
            if "dateFrom" in params and "dateTo" in params and params["dateFrom"] == params["dateTo"]:
                from datetime import timedelta
                dt = date.fromisoformat(params["dateTo"])
                params["dateTo"] = (dt + timedelta(days=1)).isoformat()
            return client.get("/ledger/posting", params=params)

        # ── Contact ──────────────────────────────────────────
        case "list_contacts":
            params = _fields_params(args)
            if "customerId" in args:
                params["customerId"] = args["customerId"]
            return client.get("/contact", params=params)

        case "create_contact":
            body = _safe_json(args["body"])
            return client.create("/contact", body)

        # ── Salary / Payroll ─────────────────────────────────
        case "list_salary_types":
            params = _fields_params(args)
            for k in ("name", "number"):
                if k in args:
                    params[k] = args[k]
            return client.get("/salary/type", params=params)

        case "create_salary_transaction":
            body = _safe_json(args["body"])
            params = {}
            if "generateTaxDeduction" in args:
                params["generateTaxDeduction"] = args["generateTaxDeduction"]
            result = client.create("/salary/transaction", body, params=params if params else None)
            if "_error" in result:
                detail = str(result.get("detail", "")).lower()
                if "arbeidsforhold" in detail:
                    # Retry with current year/month in case the period is out of range.
                    today = date.today()
                    body["year"] = today.year
                    body["month"] = today.month
                    payslips = body.get("payslips") if isinstance(body, dict) else None
                    if isinstance(payslips, list):
                        for ps in payslips:
                            if isinstance(ps, dict):
                                ps["date"] = f"{today.year}-{today.month:02d}-01"
                    result = client.create("/salary/transaction", body, params=params if params else None)

                # If it still fails, try to create an employment relationship and retry once.
                if "_error" in result:
                    detail = str(result.get("detail", "")).lower()
                    if ("arbeidsforhold" in detail or "virksomhet" in detail) and isinstance(body, dict):
                        payslips = body.get("payslips")
                        if isinstance(payslips, list) and payslips:
                            ps0 = payslips[0]
                            if isinstance(ps0, dict):
                                emp = ps0.get("employee", {})
                                employee_id = emp.get("id") if isinstance(emp, dict) else None
                                start_date = ps0.get("date")
                                if not start_date:
                                    start_date = f"{today.year}-{today.month:02d}-01"

                                if employee_id is not None:
                                    # Ensure employee has dateOfBirth (required for employment)
                                    emp_data = client.get(f"/employee/{employee_id}", params={"fields": "*"})
                                    if "_error" not in emp_data:
                                        emp_val = emp_data.get("value", emp_data)
                                        if isinstance(emp_val, dict) and not emp_val.get("dateOfBirth"):
                                            emp_val["dateOfBirth"] = "1990-01-01"
                                            client.put(f"/employee/{employee_id}", json=emp_val)

                                    # Ensure a division exists
                                    _div_id = None
                                    div_resp = client.get("/division", params={"fields": "id,name"})
                                    if "_error" not in div_resp:
                                        div_vals = div_resp.get("values", [])
                                        if isinstance(div_vals, list) and div_vals:
                                            _div_id = div_vals[0].get("id")
                                        else:
                                            # Try to create a division
                                            mun_resp = client.get("/municipality", params={"count": 1})
                                            _mun_id = None
                                            if "_error" not in mun_resp:
                                                mun_vals = mun_resp.get("values", [])
                                                if isinstance(mun_vals, list) and mun_vals:
                                                    _mun_id = mun_vals[0].get("id")
                                            if _mun_id:
                                                div_created = client.create("/division", {
                                                    "name": "Hovedvirksomhet",
                                                    "startDate": "2024-01-01",
                                                    "municipalityDate": "2024-01-01",
                                                    "municipality": {"id": _mun_id},
                                                    "organizationNumber": "000000000",
                                                })
                                                if "_error" not in div_created:
                                                    _div_id = div_created.get("id") or div_created.get("value", {}).get("id")

                                    # Check existing employments for this employee
                                    existing = client.get(
                                        "/employee/employment",
                                        params={
                                            "employeeId": employee_id,
                                            "fields": "id,version,startDate,endDate,division",
                                        },
                                    )
                                    has_employment = False
                                    existing_emp_id = None
                                    existing_emp_version = None
                                    needs_division_update = False
                                    if "_error" not in existing and isinstance(existing, dict):
                                        values = existing.get("values", [])
                                        if isinstance(values, list):
                                            for e in values:
                                                if isinstance(e, dict):
                                                    s = e.get("startDate")
                                                    end = e.get("endDate")
                                                    if s and s <= start_date and (not end or end >= start_date):
                                                        has_employment = True
                                                        existing_emp_id = e.get("id")
                                                        existing_emp_version = e.get("version", 0)
                                                        if not e.get("division") and _div_id:
                                                            needs_division_update = True
                                                        break
                                            # If no matching period, take the first one
                                            if not has_employment and values:
                                                e = values[0]
                                                has_employment = True
                                                existing_emp_id = e.get("id")
                                                existing_emp_version = e.get("version", 0)
                                                if not e.get("division") and _div_id:
                                                    needs_division_update = True

                                    if not has_employment:
                                        new_emp = {
                                            "employee": {"id": employee_id},
                                            "startDate": start_date,
                                            "isMainEmployer": True,
                                        }
                                        if _div_id:
                                            new_emp["division"] = {"id": _div_id}
                                        created = client.create("/employee/employment", new_emp)
                                        if "_error" in created and "dateOfBirth" in str(created.get("detail", "")).lower():
                                            # dateOfBirth still not set — force set it
                                            emp2 = client.get(f"/employee/{employee_id}", params={"fields": "*"})
                                            if "_error" not in emp2:
                                                ev = emp2.get("value", emp2)
                                                if isinstance(ev, dict):
                                                    ev["dateOfBirth"] = "1990-01-01"
                                                    client.put(f"/employee/{employee_id}", json=ev)
                                            created = client.create("/employee/employment", new_emp)
                                        if "_error" not in created:
                                            emp_id = created.get("id") or created.get("value", {}).get("id")
                                            if emp_id:
                                                client.create(
                                                    "/employee/employment/details",
                                                    {
                                                        "employment": {"id": emp_id},
                                                        "date": start_date,
                                                        "percentageOfFullTimeEquivalent": 100,
                                                    },
                                                )
                                    elif needs_division_update and existing_emp_id and _div_id:
                                        # Employment exists but has no division — update it
                                        full_emp = client.get(f"/employee/employment/{existing_emp_id}", params={"fields": "*"})
                                        if "_error" not in full_emp:
                                            emp_obj = full_emp.get("value", full_emp)
                                            if isinstance(emp_obj, dict):
                                                emp_obj["division"] = {"id": _div_id}
                                                client.put(f"/employee/employment/{existing_emp_id}", json=emp_obj)

                                    result = client.create("/salary/transaction", body, params=params if params else None)

            return result

        case "list_salary_payslips":
            params = _fields_params(args)
            for k in ("employeeId", "yearFrom", "yearTo", "monthFrom", "monthTo"):
                if k in args:
                    params[k] = args[k]
            return client.get("/salary/payslip", params=params)

        # ── Supplier ─────────────────────────────────────────
        case "list_supplier_invoices":
            params = _fields_params(args)
            for k in ("supplierId", "invoiceDateFrom", "invoiceDateTo"):
                if k in args:
                    params[k] = args[k]
            return client.get("/supplierInvoice", params=params)

        case "pay_supplier_invoice":
            inv_id = args["invoiceId"]
            params = {
                "paymentDate": args["paymentDate"],
                "paymentTypeId": args["paymentTypeId"],
                "paidAmount": args["paidAmount"],
            }
            return client._request("PUT", f"/supplierInvoice/{inv_id}/:registerPayment", params=params)

        case "create_supplier_invoice":
            body = _safe_json(args["body"])
            # CRITICAL: Fix voucher postings for the Tripletex API.
            # 1. Row 0 is reserved → always set row=1,2,...
            # 2. API only uses GROSS amounts: "amount" is ignored, must use "amountGross"
            if "voucher" in body and isinstance(body["voucher"], dict):
                postings = body["voucher"].get("postings", [])
                supplier_id = None
                if isinstance(body.get("supplier"), dict):
                    supplier_id = body["supplier"].get("id")
                if isinstance(postings, list):
                    for i, posting in enumerate(postings):
                        if isinstance(posting, dict):
                            posting["row"] = i + 1  # Always override row
                            if "amountGross" not in posting and "amount" in posting:
                                posting["amountGross"] = posting.pop("amount")
                            if "amountGross" in posting and "amountGrossCurrency" not in posting:
                                posting["amountGrossCurrency"] = posting["amountGross"]
                            # Auto-add supplier ref to credit posting on account 2400 (leverandørgjeld)
                            if supplier_id and "supplier" not in posting:
                                amt = posting.get("amountGross", 0)
                                if isinstance(amt, (int, float)) and amt < 0:
                                    posting["supplier"] = {"id": supplier_id}
            result = client.create("/supplierInvoice", body)
            if result.get("_error") and result.get("status_code") == 500:
                # Retry up to 4 times with exponential backoff on 500 errors.
                import time as _time
                for _retry in range(4):
                    _time.sleep(3 * (2 ** _retry))  # 3s, 6s, 12s, 24s
                    result = client.create("/supplierInvoice", body)
                    if not (result.get("_error") and result.get("status_code") == 500):
                        break
            if result.get("_error") and result.get("status_code") == 500:
                return {"_error": True, "detail": "Server error (500) on supplierInvoice creation after 5 attempts. The API may be temporarily unavailable. Try a different approach or simplify the request body."}
            return result

        case "list_suppliers":
            params = _fields_params(args)
            for k in ("name", "organizationNumber"):
                if k in args:
                    params[k] = args[k]
            return client.get("/supplier", params=params)

        case "create_supplier":
            body = {"name": args["name"], "isSupplier": True}
            for k in ("organizationNumber", "email", "phoneNumber"):
                if k in args and args[k] is not None:
                    body[k] = args[k]
            if "isCustomer" in args:
                body["isCustomer"] = args["isCustomer"]
            return client.create("/supplier", body)

        # ── Generic fallback ─────────────────────────────────
        case "tripletex_api_call":
            method = args["method"].upper()
            path = args["path"]
            params = _safe_json(args.get("params")) if args.get("params") else None
            body = _safe_json(args.get("body")) if args.get("body") else None
            if method == "PUT" and path.startswith("/project/") and isinstance(body, dict):
                body.pop("projectRateTypes", None)
                body.pop("projectratetypes", None)
            return client._request(method, path, params=params, json=body)

        case _:
            return {"_error": True, "detail": f"Unknown tool: {name}"}


# ── System prompt ────────────────────────────────────────────────────
# All facts below are verified from the OpenAPI spec and live sandbox testing.

SYSTEM_PROMPT = """\
You are an expert accounting agent for Tripletex, a Norwegian accounting system.

You receive a task prompt (in Norwegian, English, Spanish, Portuguese, Nynorsk, German, or French) and must execute it using the Tripletex v2 REST API via the provided tools.

CRITICAL RULES FOR EFFICIENCY (your score depends on this):
1. **Plan first**: Before making ANY API call, fully analyze the prompt to understand all entities, fields, and relationships needed.
2. **Minimize calls**: Every API call counts against your efficiency score. Don't fetch things you already know from previous responses.
3. **Zero errors**: Every 4xx error reduces your score. Use only verified field names from the tool descriptions.
4. **One-shot execution**: Get each call right the first time. No trial-and-error.
5. **Stop when done**: Once you have completed all required actions, stop immediately. Do not make verification GET calls unless the task requires reading data.
6. **Handle duplicates**: If a creation fails because the entity already exists (e.g. duplicate email/name), search for the existing entity and proceed with the rest of the task (e.g. grant entitlement to the found employee, use the found product). Do NOT give up.
7. **Retry on validation errors**: If a field causes a validation error, retry without that field rather than giving up. The task is still expected to be completed as best as possible.
8. **NEVER return text saying you will do something**. You MUST include ALL required tool calls in your responses. If a task has multiple steps, complete ALL steps via tool calls before returning a text-only summary. A text-only response means you are DONE — the system will stop the agent. Return text ONLY when ALL tasks are fully completed.
9. **Sequential dependencies**: When step N+1 needs data from step N (like an ID), call step N FIRST and WAIT for the result. Do NOT call both in the same turn with placeholder values. Common examples:
   - Travel expense: create header first → then add costs/per diem with the returned ID
   - Invoice flow: create order first → then create invoice with the order ID
   - Employee contract: create employee first → then employment → then employment details

VERIFIED FACTS (from OpenAPI spec and live testing):

Employee creation:
- Required fields: firstName, lastName, userType, department
- userType enum: "STANDARD" (limited access), "EXTENDED" (can receive all entitlements), "NO_ACCESS" (no login)
- CRITICAL: userType "STANDARD" and "EXTENDED" REQUIRE an email address. If no email is available, use "NO_ACCESS".
- CRITICAL: If the task provides an email address, ALWAYS use userType="EXTENDED" (not "NO_ACCESS"). 
  Only use "NO_ACCESS" when there is truly no email provided.
- department must be an object {id: <dept_id>}. Get dept_id from list_departments (there's always at least one).
- To make someone a "kontoadministrator" / administrator:
  1. Create employee with userType="EXTENDED"
  2. Call grant_entitlement with entitlementId=1 (ROLE_ADMINISTRATOR) and customerId=companyId from the created employee
- There is NO "isAdministrator" field on the employee object.
- EMPLOYMENT CONTRACT TASKS: When creating from an employment contract (arbeidskontrakt/ansettelsesavtale), you MUST:
  1. Read the PDF VERY carefully. Extract ALL fields: firstName, lastName, email, dateOfBirth, nationalIdentityNumber,
     bankAccountNumber, address, phoneNumber, start date, salary, job title (stillingstittel), work percentage.
  2. Create the employee (POST /employee) with ALL extracted fields including email, nationalIdentityNumber, dateOfBirth, bankAccountNumber.
     Use userType="EXTENDED" if email is provided.
  3. Find the occupation code: call list_occupation_codes(nameNO="<job title>").
     If exact match fails (0 results), try BROADER search terms:
     - "Seniorutvikler" → try "utvikler" or "programvareutvikler" or "systemutvikler"
     - "Regnskapssjef" → try "regnskap" or "økonomisjef"
     - "Markedsansvarlig" → try "marked" or "markedsfører"
     - Try the first word, or a more general term from the job title
     If still no results, try list_occupation_codes(count=50) and pick the closest match.
  4. Create employment: call create_employment with body {employee:{id}, startDate:"YYYY-MM-DD", isMainEmployer:true}
     - Do NOT include occupationCode on employment — it goes on employment details
  5. Create employment details: call create_employment_details with body {employment:{id}, date:"<startDate>", percentageOfFullTimeEquivalent:<pct>, annualSalary:<amount>, occupationCode:{id:<occ_id>}, remunerationType:"MONTHLY_WAGE"}
  - Use occupationCode:{id:<id>} from the list_occupation_codes result (NOT {code:"..."})
  - annualSalary is the yearly salary. If only monthly is given: annualSalary = monthly * 12
  - Do NOT use monthlyWage or paymentType — these fields DO NOT EXIST
  - remunerationType values: MONTHLY_WAGE, HOURLY_WAGE, COMMISSION, FEE, PIECEWORK_WAGE
  - Always use the employment start date from the contract
  ALL 5 STEPS ARE REQUIRED. Missing any step means losing points. Do NOT skip employment or employment details.

Customer creation:
- Required: name
- Always set isCustomer=true for customers
- Verified fields: name, email, phoneNumber, phoneNumberMobile, organizationNumber, isCustomer, isSupplier, isPrivateIndividual, language, invoiceSendMethod, invoicesDueIn, invoicesDueInType, postalAddress, physicalAddress, accountManager, department

Product creation:
- Required: name
- Price fields: priceExcludingVatCurrency, priceIncludingVatCurrency
- If the task specifies a product NUMBER, FIRST search list_products(number=X) to check if it exists. Only create if not found.
- vatType: see VAT types section below. Generally do NOT specify unless task requires a specific rate.

Invoice flow:
- Create customer → create product → create order (with orderLines) → create invoice (referencing order)
- Invoice requires: invoiceDate, invoiceDueDate, orders:[{id}]
- CRITICAL: You MUST complete the FULL flow. Creating an order is NOT enough — you MUST also create the invoice from it.
- Valid InvoiceDTO fields: id, invoiceNumber, invoiceDate, invoiceDueDate, customer, amount, amountCurrency, amountExcludingVat, amountExcludingVatCurrency, amountOutstanding, isCreditNote, orders, orderLines, voucher, isApproved
- Do NOT use 'number' (use 'invoiceNumber') or 'order' (use 'orders') as field names — these will 400
- BANK ACCOUNT: For invoice/payment tasks, account 1920 is automatically configured before your first turn. You do NOT need to set up account 1920.
  If invoice creation still fails with "bankkontonummer" error, the system will auto-retry. Do NOT manually set up bank accounts.
  For non-invoice tasks (employee, customer, project), bank setup is skipped to save API calls.

Credit notes:
- To create a credit note: find the original invoice via list_invoices (use wide date range 2020-01-01 to 2030-12-31), then call create_credit_note with the invoice ID and date
- The credit note date MUST be on or after the invoiceDate. If you get a "Dato kan ikke være før fakturadato" error, retry with the invoiceDate.
- If list_invoices for the customer returns empty, run list_invoices without customerId and match by amountExcludingVat/amount (and description if available). Never ask the user.
- This uses PUT /invoice/{id}/:createCreditNote with query params

Payment:
- For INVOICE payments (receiving money), use list_invoice_payment_types (endpoint: /invoice/paymentType)
- Do NOT use list_payment_types for invoice payments — those are OUTGOING types
- Payment registration uses QUERY PARAMETERS (not JSON body): paymentDate, paymentTypeId, paidAmount
- The amount field is called 'paidAmount' (NOT 'amount')
- Payment reversal: use tripletex_api_call PUT /invoice/{id}/:payment with NEGATIVE paidAmount (e.g. paidAmount=-5000).
  Do NOT use paidAmount=0 — this causes a 422 "amountBasisCurrency" error. Always use the negative of the original payment amount.

Order:
- Order endpoint is /order (NOT /order/order)
- Required: customer:{id}, orderDate, deliveryDate
- orderLines: product:{id}, count, unitPriceExcludingVatCurrency
- list_orders REQUIRES orderDateFrom and orderDateTo. Always use wide range: 2020-01-01 to 2030-12-31

Product lookup:
- If the task gives product NUMBERS (e.g. "product 5566"), FIRST search with list_products by number to check if they already exist. Only create if not found.
- Product numbers in Tripletex are unique — creating with an existing number will fail.

VAT types:
- Valid VatTypeDTO fields: id, name, percentage, number. Do NOT use 'rate' or 'type' — these will 400.
- If pre-fetched VAT type IDs are provided in the task context, use them directly — do NOT call list_vat_types again.
- When creating a product, do NOT specify vatType unless explicitly required.
- If you need a specific VAT rate and no pre-fetched data is available, call list_vat_types with fields=id,name,percentage.
- IMPORTANT: When calling list_vat_types, always use fields=id,name,percentage to minimize response size

Salary / Payroll:
- To find salary types: call list_salary_types (Fastlønn/Månedslønn = base salary, Timelønn = hourly, Bonus/Tillegg = supplements)
- To create a salary slip: call create_salary_transaction with year, month, and payslips array
- Each payslip: {employee:{id}, date:"YYYY-MM-01", specifications:[{salaryType:{id}, rate, count}]}
- For "this month": use current year/month. Rate = salary amount, count = 1 for monthly.
- To view existing payslips: call list_salary_payslips
- CRITICAL PREREQUISITE: Salary transactions REQUIRE the employee's employment to be linked to a division (virksomhet).
  If you get "Arbeidsforholdet er ikke knyttet mot en virksomhet" error:
  1. GET /division to find existing divisions
  2. If no divisions exist, create one: POST /division with {name, startDate, municipalityDate, municipality:{id}}
     Get municipality from GET /municipality?count=1
  3. Update the employment (PUT /employee/employment/{id}) to include division:{id}
  4. Retry the salary transaction
  If the system provides a pre-fetched division ID, use it when creating/updating employments.
- CRITICAL: The employee MUST have dateOfBirth set before creating employment. If employment creation fails
  with "dateOfBirth" required, GET the employee with fields=*, set dateOfBirth (use a reasonable date like
  1990-01-01 if not specified), then retry employment creation.
- SALARY TASK FULL FLOW: find employee → ensure dateOfBirth set → check/create division →
  create employment WITH division:{id} → create employment details → find salary types → create salary transaction

Project creation:
- Required: name, projectManager:{id}, startDate (YYYY-MM-DD)
- Optional: customer:{id}, isFixedPrice (bool), fixedprice (number — lowercase 'p'!), number
- CRITICAL: The API field is 'fixedprice' with lowercase 'p'. 'fixedPrice' (capital P) does NOT exist and will 422.
- If the task doesn't specify a start date, use today's date
- ALWAYS create the project — never ask the user for missing info, use sensible defaults

Order:
- CRITICAL: 'project' field goes on the ORDER object, NOT on orderLines. OrderLine does NOT have a 'project' field.
- Example: {customer:{id}, orderDate:'...', deliveryDate:'...', project:{id}, orderLines:[{product:{id}, count:1, unitPriceExcludingVatCurrency:1000}]}

Dates: Always YYYY-MM-DD format. IMPORTANT: February has 28 days in most years and 29 only in leap years (divisible by 4). 2025 and 2026 are NOT leap years (Feb has 28 days only). Use Feb 28 for end-of-February dates in these years.

ENDPOINTS THAT DO NOT EXIST (will 404):
- /authentications, /accessrights, /users/accessrights, /paymentType
- /travelExpense/costType (use /travelExpense/costCategory instead)

IMPORTANT API SYNTAX:
- Nested field expansion uses PARENTHESES only: fields=postings(amount,account(id,name,number)). 
  Do NOT use curly braces {} for field nesting — these cause 400 errors.
- Activity creation (POST /activity): activityType MUST be a NUMBER (integer), NOT a string.
  Valid values: 0 = GENERAL. If 0 fails, try other small integers (1, 2, 3).
  Example body: {"name": "Design", "activityType": 0}
  Do NOT use strings like "GENERAL" or "PROJECT_SPECIFIC" — these cause "Verdien er ikke av korrekt type" errors.
- Prefer using EXISTING activities from list_activities instead of creating new ones.
  Most sandboxes already have activities. Only create new ones if absolutely needed.

Ledger posting queries:
- list_postings uses /ledger/posting (NOT /ledger/posting/openPost). Requires dateFrom and dateTo.
- For analyzing general ledger data by period, use list_postings with fields=amount,account(id,name,number) and dateFrom/dateTo
- To find accounts with largest changes: query two periods separately and compare totals per account
- Expense accounts are typically in the 4000-7999 range
- LEDGER ANALYSIS TASKS: After analyzing postings, create projects for the accounts with biggest absolute changes.
  Use the ACCOUNT NAME as the project name. For timesheet entries on these projects, use EXISTING activities from list_activities.
  Do NOT try to create new activities — use existing ones.

Travel expenses:
- CRITICAL: Two types exist:
  - "reiseregning" (travel report): supports per diem AND costs. Created by including travelDetails.
  - "ansattutlegg" (employee disbursement): only supports costs. Created when travelDetails is omitted.
- CRITICAL ORDERING: Travel expense creation is MULTI-STEP WITH DEPENDENCIES:
  1. FIRST create the travel expense header and WAIT for the response to get the ID
  2. THEN add costs and/or per diem using that ID
  Do NOT call create_travel_expense_cost or create_per_diem_compensation in the same turn as create_travel_expense!
  You need the travel expense ID from step 1 before you can do step 2.
- Step 1: Create report header: POST /travelExpense with:
  {employee:{id}, title:"trip name", travelDetails:{departureDate:"2025-03-01", returnDate:"2025-03-03", departureTime:"08:00", returnTime:"18:00", departureFrom:"Oslo", destination:"Trondheim"}}
  ALWAYS include travelDetails if the task involves per diem, travel days, or overnight stays!
- Step 2a: Add cost lines: POST /travelExpense/cost with {travelExpense:{id}, amountCurrencyIncVat, paymentType:{id}}
  - Get paymentType IDs from /travelExpense/paymentType (NOT /ledger/paymentTypeOut)
  - Optional: costCategory:{id}, date, category (string), comments, isPaidByEmployee
- Step 2b: Add per diem: POST /travelExpense/perDiemCompensation with {travelExpense:{id}, location, count (days)}
  - Per diem ONLY works on "reiseregning" type (with departure/return dates set)
- Per diem supports 'rate' field (e.g. rate:800 for 800 NOK/day) and 'count' (number of days)
  If the task specifies a daily rate, include it: {travelExpense:{id}, location:"Oslo", count:3, rate:800}
- Cost categories from /travelExpense/costCategory

PUT operations: You MUST include ALL fields including 'id' and 'version' from the GET response.

Voucher (journal entry / bilag):
- Required: date (YYYY-MM-DD), description, postings (array)
- CRITICAL: The API ONLY uses 'amountGross' for postings. 'amount' is IGNORED. Always use 'amountGross'.
- CRITICAL: Each posting MUST include "row" field starting from 1 (row 0 is reserved for system-generated postings and will cause a 422 error)
- CRITICAL: Postings on account 1500 (Kundefordringer) MUST include customer:{id}. Postings on account 2400 (Leverandørgjeld) MUST include supplier:{id}. Missing these causes a validation error.
- Each posting: {row:<N>, account:{id}, amountGross (positive=debit, negative=credit), description}
- Postings can reference: department:{id}, project:{id}, product:{id}, customer:{id}, supplier:{id}
- Voucher postings MUST balance (sum of amountGross = 0)
- To find account IDs: use list_accounts with number parameter (e.g. account 6340)
- Example: {date:"2025-03-01", description:"Expense", postings:[{row:1, account:{id:X}, amountGross:1000, department:{id:Y}}, {row:2, account:{id:Z}, amountGross:-1000}]}

Custom dimensions (fri rekneskapsdimensjon / Marked etc.):
- Tripletex uses departments and projects as accounting dimensions
- If the task asks to create a "dimension" with values, create DEPARTMENTS for each value
- Then when posting vouchers, attach the department to the posting
- Example: "Create dimension 'Marked' with 'Offentlig' and 'Privat'" → create departments named "Offentlig" and "Privat"

Supplier invoices (leverandørfaktura) AND receipt/voucher booking (kvittering):
- BOTH supplier invoices AND receipt bookings use create_supplier_invoice tool (POST /supplierInvoice)
- When a task says "book this receipt" (bokfør kvittering) or similar, use create_supplier_invoice — NOT a standalone voucher
- Required: invoiceDate, supplier:{id}
- CRITICAL: Account assignment goes via voucher.postings, NOT via orderLines!
- CRITICAL: Each posting MUST include "row" field starting from 1 (same rule as vouchers)
- CRITICAL: Use 'amountGross' NOT 'amount' — the API only uses gross amounts. 'amount' is IGNORED!
- CRITICAL: You MUST include BOTH the debit posting (expense account, positive amountGross) AND the credit posting (supplier account 2400, negative amountGross). Missing the credit posting causes "credit posting missing" error.
- CRITICAL: The credit posting on account 2400 MUST include supplier:{id} — without it, you get 500 errors!
- If PRE-FETCHED data provides account 2400 id and input VAT id, use them directly — do NOT look them up again!
- AMOUNT RULES WITH VAT: When the debit posting has a vatType set (e.g. 25% input VAT), Tripletex adds VAT ON TOP of amountGross.
  Example with 25% VAT: debit amountGross=40600, vatType=25% → Tripletex books expense 40600 + VAT 10150 = 50750 total.
  So: debit amountGross = amount EXCLUDING VAT (net). Credit amountGross = -(amount INCLUDING VAT) = -(net * 1.25).
  Formula: debit_gross = total_incl_vat / 1.25, credit_gross = -total_incl_vat
- Supplier account (leverandørgjeld): typically account 2400 — use pre-fetched ID if available.
- Input VAT type: "Fradrag inngående avgift, høy sats" (25%) — use pre-fetched ID if available.
- To find/create suppliers: use list_suppliers or create_supplier (NOT create_customer with isSupplier)

PDF/Receipt reading — CRITICAL:
- Read ALL fields from the PDF VERY carefully: supplier name, org number, invoice number, dates, line item descriptions, amounts, account numbers, VAT details
- The PDF specifies the EXACT expense account number to use (e.g. "Konto: 6540"). Use THAT account number — do NOT guess.
- Common Norwegian expense accounts: 6300=Leie lokale, 6340=Lys/varme, 6500=Verktøy, 6540=Inventar/utstyr, 6800=Kontorrekvisita, 6860=IT-utstyr
- The invoice number from the PDF (e.g. "INV-2026-1234") MUST be set as invoiceNumber on the supplier invoice
- Use the EXACT dates from the PDF: invoiceDate, invoiceDueDate (forfallsdato)
- Include ALL line items from the PDF, not just the first one

Timesheet / Hour logging:
- Use create_timesheet_entry tool (POST /timesheet/entry)
- Required: employee:{id}, activity:{id}, date, hours
- Optional: project:{id}, comment
- First find the activity ID with list_activities(name="Design") — activities are system-wide, not per-project
- Date MUST be >= the project's startDate. If you get a "Startdato" validation error:
  1. Parse the project start date from the error message
  2. Use that date (or today's date, whichever is later) and retry
  3. NEVER abandon timesheet logging—the grading checks for actual timesheet entries!

Bank reconciliation (bankavstemming):
- Parse the CSV file to get ALL transactions (date, description, amount, reference)
- STEP 1: Get all customers and suppliers in ONE call each:
  list_customers(fields="id,name", count=100) — get ALL customers
  list_suppliers(fields="id,name", count=100) — get ALL suppliers
  Do NOT search by individual name — this returns fuzzy matches of all entities!
- STEP 2: Get ALL invoices and supplier invoices:
  list_invoices(invoiceDateFrom="2020-01-01", invoiceDateTo="2030-12-31", fields="id,invoiceNumber,amount,amountOutstanding,customer", count=100)
  list_supplier_invoices(invoiceDateFrom="2020-01-01", invoiceDateTo="2030-12-31", fields="id,invoiceNumber,amount,supplier(id,name)", count=100)
- STEP 3: Match CSV transaction descriptions to customers/suppliers by comparing the names.
  Match invoice references (numbers) from the CSV to invoice numbers.
- STEP 4: For each transaction:
  - INCOMING payments (positive amounts for customer payments): Find the matching customer invoice then register with create_payment.
    Use the invoice payment type "Betalt til bank" for all incoming payments.
  - OUTGOING payments (negative amounts for supplier payments): Find the matching supplier invoice via list_supplier_invoices,
    then register payment using pay_supplier_invoice(invoiceId=X, paymentDate="YYYY-MM-DD", paymentTypeId=Y, paidAmount=<amount>).
    The paidAmount for supplier payments should be the ABSOLUTE value (positive, not negative).
    Use list_payment_types (GET /ledger/paymentTypeOut) to get outgoing payment type IDs.
    Choose "Betaling fra bank" or the bank transfer type.
    CRITICAL: Do NOT create manual vouchers for supplier payments — use pay_supplier_invoice!
    The grading specifically checks that supplier invoice payments are registered properly.
  - Bank fees (bankgebyr): Create voucher debit 7770, credit 1920
  - Tax/public payments (skatt, skattetrekk, MVA): Create voucher debit 2600 (Skattetrekk), credit 1920
- CRITICAL: Handle ALL transactions from the CSV. Missing even one transaction loses points.
- PARTIAL PAYMENTS: If a bank amount < invoice total, register the partial amount.
- Keep the payment DATE from the CSV for each transaction.

Currency exchange rate / Agio-Disagio tasks:
- When an invoice was sent in foreign currency and the customer pays at a different exchange rate:
  1. Find the invoice (list_invoices) — note the original amount in both currencies
  2. Calculate: original_nok = foreign_amount * original_rate, payment_nok = foreign_amount * payment_rate
  3. Register payment with create_payment: paidAmount = payment_nok (what was actually received in NOK)
  4. The exchange rate DIFFERENCE is booked as agio (gain) or disagio (loss):
     - If payment_rate > original_rate → gain (agio): credit account 8060 (Valutagevinst)
     - If payment_rate < original_rate → loss (disagio): debit account 8160 (Valutatap)
  5. Create a voucher for the exchange rate difference:
     - Gain: debit 1500 (Kundefordringer) for the difference, credit 8060 (Valutagevinst)
     - Loss: debit 8160 (Valutatap), credit 1500 (Kundefordringer)
     - The difference = abs(payment_nok - original_nok)
  6. IMPORTANT: The invoice payment (step 3) may handle the exchange difference automatically
     via Tripletex. Check the voucher it creates. If the exchange difference is already booked, do NOT double-book it.
  7. For supplier invoices in foreign currency: same logic but with account 2400 and pay_supplier_invoice.

IMPORTANT BEHAVIORS:
- NEVER ask the user for more information. You must complete tasks with the data provided, using sensible defaults for anything missing.
- If something fails, try to fix it (e.g., bank account error → set bank account, duplicate product → search existing).
- list_invoices ALWAYS needs invoiceDateFrom and invoiceDateTo. Use 2020-01-01 to 2030-12-31 as default.
- list_orders ALWAYS needs orderDateFrom and orderDateTo. Use 2020-01-01 to 2030-12-31 as default.
- IMPORTANT: dateTo on list_vouchers/list_postings is EXCLUSIVE. For a single date use dateFrom=DATE, dateTo=DATE+1day. We auto-fix this but prefer correct dates.
- 'isClosed' is NOT a valid filter on invoices or supplier invoices. Do NOT use it.
- For dates not specified in the task, use today's date.
- For project number: OMIT the number field unless the task specifies one. Tripletex auto-assigns project numbers.
- For project start date not specified, use today's date.

ERROR RECOVERY RULES:
- Date validation: If a date is rejected (e.g. "Startdato"), extract the valid date from the error and retry with that date.
- Unknown field: If you get "Feltet eksisterer ikke" (field doesn't exist), you have the wrong schema. Switch to the correct field names documented in the tool descriptions.
- Bank account: Always use "12345678903" (valid Norwegian MOD-11 bank number). Do NOT make up bank numbers.
- 403/401 errors: The token may have expired. Stop making API calls and return what you completed.
- NEVER abandon the primary requested action in favor of a workaround. The grading checks for the specifically requested entity type.
- Account 2400 (Leverandørgjeld): EVERY posting on this account MUST include supplier:{id}. If the voucher is a correction,
  first find the original supplier from the original posting (list_postings or get the voucher). If no supplier exists, create one.
  Switching to a different account (like 1920) to avoid the supplier requirement is WRONG — the grading checks for account 2400.

ERROR CORRECTION / JOURNAL ENTRY TASKS:
- When asked to correct a booking/voucher error, create a NEW correction voucher (bilag) that reverses the wrong entries and books the correct ones.
- STEP 1: Get all vouchers/postings. Use a WIDE date range (dateFrom="2020-01-01", dateTo="2030-12-31") to be safe. 
  NEVER use the exact same date for dateFrom and dateTo — dateTo is exclusive. For a specific date X, use dateTo = X + 1 day.
- STEP 2: Identify the errors described in the task by examining the postings.
- STEP 3: Create ONE correction voucher that reverses the wrong entries and books the correct ones.
- When moving amounts between accounts, ALWAYS preserve the original dimensions: supplier:{id} on 2400, customer:{id} on 1500.
- If the original voucher has a supplier on account 2400, your correction voucher MUST also reference that same supplier on any 2400 postings.
- CRITICAL: Voucher postings MUST balance: sum of all amountGross values = 0. Double-check your arithmetic!
  For reversal: negate EVERY posting from the original (debit becomes credit, credit becomes debit).
  Then add correct postings. Keep reversal and correction in the SAME voucher if possible.
- CRITICAL: Do NOT use vatType on correction vouchers unless the original used it. VAT changes the effective amount!
  If the original posting was on an expense account with VAT, the reversal must use the same vatType so the
  system-generated VAT posting also reverses. Or use vatType with id=0 (no VAT) and manually handle amounts.
- EFFICIENCY: Read the existing vouchers ONCE (list vouchers with WIDE date range), extract the wrong postings,
  then create ONE correction voucher. Do NOT make multiple correction vouchers.

When you receive a task:
1. What language is the prompt in? Understand it.
2. What entities need to be created/modified/deleted?
3. What are the exact field values from the prompt?
4. What prerequisites exist or need to be created?
5. What is the minimum sequence of API calls?
6. Execute the plan. Stop when done. NEVER stop to ask questions.
"""


# ── Agent loop ───────────────────────────────────────────────────────

async def run_agent(
    prompt: str,
    files: list[dict],
    tripletex_client: TripletexClient,
    max_turns: int = 25,
) -> None:
    """
    Run the Gemini agent loop until the task is complete or max_turns is reached.
    """
    contents = []

    # Build user message parts
    user_parts = [types.Part.from_text(text=f"Please complete this accounting task:\n\n{prompt}")]

    # Add file contents if present
    for f in files:
        data = base64.b64decode(f["content_base64"])
        mime = f.get("mime_type", "application/octet-stream")
        if mime.startswith("image/") or mime == "application/pdf":
            user_parts.append(types.Part.from_bytes(data=data, mime_type=mime))
            user_parts.append(types.Part.from_text(text=f"[Attached file: {f['filename']}]"))
        else:
            try:
                text_content = data.decode("utf-8")
                user_parts.append(types.Part.from_text(
                    text=f"[File: {f['filename']}]\n{text_content}"
                ))
            except UnicodeDecodeError:
                user_parts.append(types.Part.from_bytes(data=data, mime_type=mime))

    # ── Detect task type from prompt to optimize pre-task calls ──────
    prompt_lower = prompt.lower()
    # Invoice/payment keywords in all 7 supported languages
    _INVOICE_KEYWORDS = [
        "invoice", "faktura", "rechnung", "factura", "fatura", "facture",
        "payment", "betaling", "zahlung", "pago", "pagamento", "paiement",
        "credit note", "kreditnota", "kredittnotat", "gutschrift", "avoir",
        "supplier invoice", "leverandørfaktura", "leverandorfaktura", "lieferantenrechnung",
        "bank", "reconcil", "avstemming", "purring", "reminder", "rappel",
        "month-end", "månedsslutt", "periode", "periodeslutt", "monthsslutt",
        "monatsende", "mahnung", "erinnerung",
        "kvittering", "receipt", "quittung", "recibo", "reçu",
        "bokfør", "bokfor", "bokført", "bokfort", "buchen",
    ]
    needs_bank = any(kw in prompt_lower for kw in _INVOICE_KEYWORDS)
    needs_supplier_context = any(kw in prompt_lower for kw in [
        "supplier invoice", "leverandørfaktura", "leverandorfaktura",
        "lieferantenrechnung", "factura de proveedor", "fatura fornecedor",
        "facture fournisseur",
        # Also trigger for receipt/voucher booking tasks (they use supplier invoices too)
        "kvittering", "receipt", "quittung", "recibo", "recibido", "reçu",
        "bokfør", "bokfor", "bokført", "bokfort", "buchen", "registrer",
    ])
    _SALARY_KEYWORDS = [
        "salary", "lønn", "lonn", "payroll", "lønnskjøring", "lonnskjoring",
        "nómina", "nomina", "salário", "salario", "gehalt", "salaire",
        "payslip", "lønnsslipp", "lonnsslipp", "lønnstransaksjon",
    ]
    needs_salary_division = any(kw in prompt_lower for kw in _SALARY_KEYWORDS)

    # ── Pre-task setup: conditional based on task type ───────────────
    prefetch_context = []  # Lines of context to inject into user message

    if needs_bank:
        # Configure bank account 1920 — only for tasks that involve invoices/payments
        try:
            acct_resp = tripletex_client.get(
                "/ledger/account",
                params={"number": "1920", "fields": "id,version,name,number,bankAccountNumber"},
            )
            if "_error" not in acct_resp:
                values = acct_resp.get("values", [])
                if values and isinstance(values, list):
                    acc = values[0]
                    bank_num = acc.get("bankAccountNumber") or ""
                    if not bank_num.strip():
                        acc_id = acc.get("id")
                        acc_version = acc.get("version", 0)
                        tripletex_client.put(
                            f"/ledger/account/{acc_id}",
                            json={
                                "id": acc_id,
                                "version": acc_version,
                                "name": acc.get("name", "Bankinnskudd"),
                                "number": acc.get("number", 1920),
                                "bankAccountNumber": "12345678903",
                                "isBankAccount": True,
                            },
                        )
                        logger.info("Pre-task: bank account 1920 configured")
                    else:
                        logger.info("Pre-task: bank account 1920 already configured")
        except Exception as e:
            logger.warning(f"Pre-task bank setup failed (non-fatal): {e}")

        # Pre-fetch account 2400 and common VAT types for supplier invoice tasks
        if needs_supplier_context:
            try:
                acct2400 = tripletex_client.get(
                    "/ledger/account",
                    params={"number": "2400", "fields": "id,number"},
                )
                if "_error" not in acct2400:
                    vals = acct2400.get("values", [])
                    if vals and isinstance(vals, list):
                        aid = vals[0].get("id")
                        prefetch_context.append(f"PRE-FETCHED: Account 2400 (Leverandørgjeld) has id={aid}")

                vat_resp = tripletex_client.get(
                    "/ledger/vatType",
                    params={"fields": "id,name,percentage"},
                )
                if "_error" not in vat_resp:
                    vals = vat_resp.get("values", [])
                    if isinstance(vals, list):
                        # Extract key VAT types
                        for v in vals:
                            name = (v.get("name") or "").lower()
                            vid = v.get("id")
                            pct = v.get("percentage")
                            if "fradrag" in name and "inngående" in name and "høy" in name:
                                prefetch_context.append(f"PRE-FETCHED: Input VAT 25% (Fradrag inngående avgift, høy sats) has id={vid}")
                            elif "utgående" in name and "høy" in name and pct == 25:
                                prefetch_context.append(f"PRE-FETCHED: Output VAT 25% (Utgående avgift, høy sats) has id={vid}")
                            elif "middels" in name and pct == 15:
                                prefetch_context.append(f"PRE-FETCHED: VAT 15% (middels sats) has id={vid}")
                logger.info(f"Pre-task: supplier context fetched ({len(prefetch_context)} items)")
            except Exception as e:
                logger.warning(f"Pre-task supplier context fetch failed (non-fatal): {e}")
    else:
        logger.info("Pre-task: skipping bank setup (not an invoice/payment task)")

    # ── Pre-task: Division setup for salary tasks ────────────────────
    division_id = None
    if needs_salary_division:
        try:
            div_resp = tripletex_client.get("/division", params={"fields": "id,name"})
            if "_error" not in div_resp:
                vals = div_resp.get("values", [])
                if isinstance(vals, list) and vals:
                    division_id = vals[0].get("id")
                    prefetch_context.append(f"PRE-FETCHED: Division (virksomhet) id={division_id}")
                    logger.info(f"Pre-task: found existing division id={division_id}")
                else:
                    # No divisions exist — try to create one
                    # Get a municipality
                    mun_resp = tripletex_client.get("/municipality", params={"count": 1})
                    mun_id = None
                    if "_error" not in mun_resp:
                        mun_vals = mun_resp.get("values", [])
                        if isinstance(mun_vals, list) and mun_vals:
                            mun_id = mun_vals[0].get("id")

                    if mun_id:
                        today_str = date.today().isoformat()
                        div_body = {
                            "name": "Hovedvirksomhet",
                            "startDate": "2024-01-01",
                            "municipalityDate": "2024-01-01",
                            "municipality": {"id": mun_id},
                            "organizationNumber": "000000000",
                        }
                        # Use a different org number than company (sub-unit)
                        div_result = tripletex_client.create("/division", div_body)
                        if "_error" not in div_result:
                            division_id = div_result.get("id") or div_result.get("value", {}).get("id")
                            if division_id:
                                prefetch_context.append(f"PRE-FETCHED: Division (virksomhet) id={division_id}")
                                logger.info(f"Pre-task: created division id={division_id}")
                        else:
                            logger.warning(f"Pre-task: division creation failed: {div_result}")
                    else:
                        logger.warning("Pre-task: could not find municipality for division")
            logger.info(f"Pre-task: salary division setup complete (division_id={division_id})")
        except Exception as e:
            logger.warning(f"Pre-task salary division setup failed (non-fatal): {e}")

    # Inject pre-fetched context into user message
    if prefetch_context:
        context_text = "\n".join(prefetch_context)
        user_parts.append(types.Part.from_text(
            text=f"\n[SYSTEM PRE-FETCHED DATA — use these IDs directly, do NOT re-fetch them]\n{context_text}"
        ))

    contents.append(types.Content(role="user", parts=user_parts))

    # Initialize Gemini client (Google AI API key for 3.1 Pro access)
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required. Set it in .env or as a Cloud Run env var.")
    gemini_client = genai.Client(api_key=gemini_api_key)

    # Agent loop
    recent_errors = []  # Track recent error messages for loop detection
    consecutive_empty = 0  # Track consecutive empty responses
    for turn in range(max_turns):
        logger.info(f"=== Agent turn {turn + 1}/{max_turns} ===")

        # Retry loop for transient Gemini errors (429 rate limit, 503 overload)
        response = None
        for attempt in range(7):
            try:
                response = gemini_client.models.generate_content(
                    model="gemini-3.1-pro-preview",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=TRIPLETEX_TOOLS,
                        temperature=0.0,
                        max_output_tokens=16384,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=24576,
                        ),
                    ),
                )
                break  # Success
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str or "UNAVAILABLE" in err_str:
                    import random
                    base_wait = min(5 * (2 ** attempt), 120)  # 5, 10, 20, 40, 80, 120, 120
                    jitter = random.uniform(0, base_wait * 0.3)
                    wait = base_wait + jitter
                    logger.warning(f"Gemini rate limit/overload (attempt {attempt+1}/7), waiting {wait:.0f}s: {err_str[:200]}")
                    await asyncio.sleep(wait)
                else:
                    logger.exception(f"Non-retryable Gemini error: {err_str[:300]}")
                    raise
        if response is None:
            logger.error("All 7 Gemini retries exhausted, aborting agent")
            return

        # Log token usage for cost tracking
        um = response.usage_metadata
        if um:
            logger.info(
                f"Tokens: in={um.prompt_token_count} out={um.candidates_token_count} "
                f"think={um.thoughts_token_count} total={um.total_token_count}"
            )

        candidate = response.candidates[0]
        content = candidate.content
        contents.append(content)

        # Guard against empty content (Gemini sometimes returns None parts)
        if content.parts is None or len(content.parts) == 0:
            consecutive_empty += 1
            logger.warning(f"Empty response from Gemini on turn {turn + 1} (consecutive: {consecutive_empty}), retrying...")
            # Remove the empty content and retry
            contents.pop()
            if consecutive_empty >= 2:
                # After 2 consecutive empties, inject a nudge to unstick the model
                nudge = types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        "The model returned empty responses. You MUST either: "
                        "1) Call a tool to take the next action, OR "
                        "2) Return your final text answer summarizing what was accomplished. "
                        "Do NOT return empty. If you're stuck, summarize what you did so far and finish."
                    )]
                )
                contents.append(nudge)
                logger.warning(f"Injected empty-response nudge after {consecutive_empty} empties")
                consecutive_empty = 0
            continue

        # Got a non-empty response — reset empty counter
        consecutive_empty = 0

        # Check if there are function calls (skip thought/thinking parts)
        function_calls = [
            part for part in content.parts
            if part.function_call is not None
        ]

        if not function_calls:
            text_parts = [p.text for p in content.parts if p.text and not getattr(p, 'thought', False)]
            full_text = " ".join(text_parts) if text_parts else ""
            if text_parts:
                logger.info(f"Agent final message: {full_text[:500]}")

            # Check if the model wants to continue but forgot to make tool calls
            _continuation_words = [
                "will proceed", "will now", "procedeé", "procederé", "voy a",
                "ahora", "next step", "next i", "now i will", "i'll now",
                "let me now", "now let me", "agora", "maintenant", "jetzt",
                "próximo paso", "prochain", "nächster",
            ]
            if full_text and any(cw in full_text.lower() for cw in _continuation_words):
                logger.warning("Agent returned text indicating more work needed — prompting continuation")
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text="You said you would continue but did not make any tool calls. "
                             "Please complete ALL remaining steps now using tool calls. "
                             "Do NOT return text describing what you will do — just call the tools."
                    )],
                ))
                continue  # Don't terminate, let the agent continue

            logger.info(f"Agent completed in {turn + 1} turns, {tripletex_client.call_count} API calls, {tripletex_client.error_count} errors")
            return

        # Execute all function calls and collect results
        function_response_parts = []
        for fc in function_calls:
            fn_name = fc.function_call.name
            fn_args = dict(fc.function_call.args) if fc.function_call.args else {}

            logger.info(f"Tool call: {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:300]})")

            try:
                result = execute_tool(tripletex_client, fn_name, fn_args)
            except Exception as e:
                logger.exception(f"Error executing {fn_name}")
                result = {"_error": True, "detail": str(e)}

            # Truncate large results to keep context manageable
            result_str = json.dumps(result, ensure_ascii=False, default=str)
            if len(result_str) > 8000:
                # For list responses, keep the structure but limit items
                if isinstance(result, dict) and "values" in result and isinstance(result["values"], list):
                    values = result["values"]
                    # Keep trimming values until under limit
                    while len(json.dumps(result, ensure_ascii=False, default=str)) > 8000 and len(values) > 5:
                        values = values[:len(values) // 2]
                        result["values"] = values
                        result["_note"] = f"Truncated to {len(values)} of {result.get('fullResultSize', '?')} results"
                else:
                    try:
                        result_str = result_str[:8000]
                        result = json.loads(result_str.rsplit(",", 1)[0] + "]}")
                    except Exception:
                        result = {"_truncated": True, "preview": result_str[:4000]}

            logger.info(f"  Result: {json.dumps(result, ensure_ascii=False, default=str)[:300]}")

            if result.get("_error") and result.get("status_code") == 403:
                detail = str(result.get("detail", "")).lower()
                if "invalid or expired proxy token" in detail:
                    logger.warning("Proxy token invalid/expired; aborting task early")
                    return

            # Track errors for loop detection
            if result.get("_error"):
                err_sig = str(result.get("detail", ""))[:200]
                recent_errors.append(err_sig)

            function_response_parts.append(
                types.Part.from_function_response(
                    name=fn_name,
                    response=result,
                )
            )

        contents.append(types.Content(role="user", parts=function_response_parts))

        # Repeated error detection: if last 3+ errors are the same, inject a nudge
        if len(recent_errors) >= 3 and len(set(recent_errors[-3:])) == 1:
            nudge = (
                f"WARNING: You have received the SAME error 3+ times in a row: {recent_errors[-1][:150]}. "
                f"You MUST change your approach immediately. Consider: "
                f"1) Are you missing a prerequisite step (e.g. division for salary, dateOfBirth for employment)? "
                f"2) Are your amounts/calculations wrong (e.g. voucher not balancing to 0)? "
                f"3) Is there a different API endpoint or field name you should use? "
                f"Do NOT repeat the same failing call."
            )
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=nudge)],
            ))
            logger.warning(f"Injected repeated-error nudge after {len(recent_errors)} errors")
            recent_errors.clear()  # Reset to avoid spamming

    logger.warning(f"Agent hit max turns ({max_turns})")
