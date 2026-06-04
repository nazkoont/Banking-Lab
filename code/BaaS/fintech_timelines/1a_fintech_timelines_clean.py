"""
1a_fintech_timelines_clean.py
=============================
Clean and consolidate raw Claude timeline batches into a master dataset.
Remaps subcategories, flags rebrands/features/variants/excluded/non-US,
applies manual corrections, follows rebrand chains, detects discontinuations,
and builds a yearly product panel.

Output:
  Data_cleaned/fintech_timelines_master.csv    Master dataset with all flags and corrections
  Data_cleaned/fintech_product_panel.csv       Yearly panel (one row per product-year, 2005-present)
"""

import pandas as pd
import glob
import os
import re
from collections import Counter

# Ensure working directory is the project root (parent of Code/)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

DATA_CLEANED_DIR = 'Data_cleaned'
CLAUDE_RAW_DIR = 'Claude_raw'
os.makedirs(DATA_CLEANED_DIR, exist_ok=True)

# =============================================================================
# 1. Load and append all batch files from Claude_raw/
# =============================================================================
files = sorted(glob.glob(os.path.join(CLAUDE_RAW_DIR, 'fintech_timeline_batch*.csv')))
dfs = [pd.read_csv(f, on_bad_lines='skip') for f in files]
df = pd.concat(dfs, ignore_index=True)

print(f"Loaded {len(df)} rows across {df['company_name'].nunique()} companies "
      f"from {len(files)} files")

# =============================================================================
# 2. Consolidate subcategories and define 3-tier taxonomy
# =============================================================================

# Remap variant subcategories to canonical names before classification
SUBCAT_REMAP = {
    # Depository → Banking (Consumer)
    "Debit Card": "Banking (Consumer)",
    "Banking (Deposit Accounts)": "Banking (Consumer)",
    "Savings/Deposits": "Banking (Consumer)",
    "Savings Account": "Banking (Consumer)",
    "Savings & Rewards": "Banking (Consumer)",
    "Prepaid Cards": "Banking (Consumer)",
    "Prepaid Card": "Banking (Consumer)",
    "Payroll Card": "Banking (Consumer)",
    "Banking (Commercial)": "Banking (Consumer)",
    # Payments cluster
    "Card Reader/Terminal": "Point-of-Sale",
    "Payments & Transfers": "Money Transfer",
    # Lending cluster
    "Lending": "Lending (Consumer)",
    "Lending & Credit": "Lending (Consumer)",
    "Cash Advance": "Lending (Consumer)",
    "Earned Wage Access": "Lending (Consumer)",
    "Credit Monitoring": "Credit Building",
    # Investing cluster
    "Robo-Advisory": "Investing",
    "Wealth Management": "Investing",
    "Investment Management": "Investing",
    "Asset Management": "Investing",
    "Brokerage": "Investing",
    "Token/ICO Platform": "Crypto/Digital Assets",
    "Venture Funds": "Alternative Investments",
    "Syndicate Investing": "Alternative Investments",
    "Crowdfunding": "Alternative Investments",
    "Capital Raising Tools": "Alternative Investments",
    "Fund Administration": "Alternative Investments",
    # Insurance & Benefits
    "Tax-Advantaged Accounts": "Insurance & Benefits",
    "Insurance": "Insurance & Benefits",
    # Other Financial Services
    "Tax Services": "Other Financial Services",
    "Expense Management": "Other Financial Services",
    "Rewards & Loyalty": "Other Financial Services",
    "Debt Resolution": "Other Financial Services",
}

# Apply remap
df['product_subcategory'] = df['product_subcategory'].replace(SUBCAT_REMAP)

# ── Flag rebrands, features, and variants ────────────────────────────────────
df['is_rebrand'] = False
df['is_feature'] = False
df['is_variant'] = False
df['is_excluded'] = False
df['is_non_us'] = False

REBRANDS = [
    # Batch 1
    ('Achieve', 'Achieve (Rebrand from Freedom Financial Network)'),
    ('Target', 'Target Check Card (Debit Card)'),
    ('Earnin', 'Activehours'),
    ('Edenred Pay', 'Corporate Spending Innovations (CSI)'),
    ('Edenred Pay', 'CSI Enterprises'),
    ('Edenred Pay', 'CSI globalVCard'),
    ('Edenred Pay', 'CSI Edenred'),
    ('Greenphire', 'eClinicalGPS'),
    ('Greenphire', 'ClinCard'),
    ('GasBuddy', 'Pay with GasBuddy (Original Fuel Savings Card)'),
    ('APPS', 'Stax Processing'),
    ('Card.com', 'Cardlike'),
    ('Card.com', 'Card.com (company rebrand)'),
    ('Brightwell', 'Prepaid Solutions (division of West Suburban Bank)'),
    ('Crypto.com', 'MCO Visa Card'),
    ('Defyne Payments', 'Defyne Processing Credit Card Processing'),
    ('Intuit', 'Intuit Merchant Services'),
    ('Heartland Payment Solutions', 'Global Payments Rebrand (from Heartland)'),
    ('Kurensē', 'iPayroll Prepaid MasterCard'),
    ('Kasheesh', 'Kasheesh 2.0'),
    ('Kasheesh', 'Kasheesh Single-Use Card (Chrome Extension)'),
    ('Global Payments', 'Heartland Payment Processing'),
    ('Merchant E-Solutions', 'Merchant e-Solutions Payment Processing'),
    ('MoCaFi', 'Akimbo MoCaFi Prepaid Mastercard'),
    ('Payment Depot', 'Payment Depot by Stax (Rebrand)'),
    ('PLS', 'PLS Visa Prepaid Card'),
    # Shift4 chain
    ('Shift4', 'Lighthouse Network'),
    ('Shift4', 'Harbortouch'),
    ('Shift4', 'United Bank Card'),
    # Shopify Pay → Shop Pay
    ('Shopify', 'Shopify Pay'),
    # SignaPay
    ('SignaPay', 'PayLo Dual Pricing'),
    ('Simpro', 'Simpro Payments (legacy Stripe integration)'),
    ('Skrill', 'Moneybookers Prepaid Mastercard'),
    ('Stax', 'Fattmerchant Payment Processing'),
    ('ThinkPoint Financial', 'ThinkPoint Financial by Deluxe'),
    ('Truss Payments', 'MazumaGo B2B Payment Platform'),
    ('Truss Payments', 'MazumaGo (Y Combinator, US Expansion)'),
    ('Vecter Technologies', 'LIFT Financial Services'),
    ('WEX', 'WEX Inc. (Rebrand from Wright Express)'),
    # Lending Consumer batch 1
    ('Achieve', 'FreedomPlus'),
    ('Achieve', 'Lendage HELOCs'),
    ('Acima', 'Simple Finance'),
    ('Acima', 'Acima Credit'),
    ('Acima', 'AcceptanceNOW (rebranded to Acima)'),
    ('Afterpay', 'Cash App Afterpay (US Rebrand)'),
    ('Ally Bank', 'GMAC Mortgage'),
    ('Avant', 'AvantCredit (Personal Loans)'),
    ('Block', 'Square Cash'),
    ('Changed', 'ChangEd Round-Up App'),
    ('Cleo', 'Cleo Cover (Salary Advance)'),
    ('Concora Credit', 'Genesis Financial Solutions'),
    ('DigniFi', 'Confident Financial Solutions Installment Loans'),
    ('EasyPay', 'Duvera Billing Services (Consumer Financing Platform)'),
    ('CreditStrong', 'Credit Builder Bank Accounts (Predecessor)'),
    ('FuturePay', 'FuturePay Revolving Credit Platform'),
    ('Grit Financial', 'Advance (Standalone EWA)'),
    ('Helix Financial', 'Helix Loans'),
    ('HFD', 'HFD (Rebrand)'),
    ('Laurel Road', 'DRB Student Loan'),
    ('LightStream', 'FirstAgain (Online Consumer Lending Platform)'),
    ('Lower', 'Homeside Financial'),
    ('OneMain Financial', 'Commercial Credit Company'),
    ('OneMain Financial', 'OneMain Financial (rebrand from CitiFinancial)'),
    ('OneMain Financial', 'CitiFinancial'),
    ('OneMain Financial', 'OneMain Holdings (rebrand from Springleaf)'),
    ('Paypal', 'Bill Me Later Acquisition (became PayPal Credit)'),
    ('Reach Financial', 'Liberty Lending Personal Loans'),
    ('Rocket', 'Quicken Loans Mortgage Services (QLMS)'),
    ('Rocket', 'Rock Financial'),
    ('Rocket', 'Quicken Loans'),
    ('Rocket', 'RockLoans.com'),
    ('Scratchpay', 'Take 5'),
    ('Stride Funding', 'AlmaPact ISA Program Management'),
    ('TrueConnect', 'TrueConnect No Credit Check Loan'),
    ('AAA', 'AAA Visa Credit Card (Bank of America)'),
    ('Amazon', 'Amazon Rewards Visa Card'),
    ('Amazon', 'Amazon Prime Rewards Visa Signature Card'),
    ('Atlas', 'Point Card Titan (Announced)'),
    ('Aven', 'Aven Home Card'),
    ('Best Buy', 'Best Buy Rewards Credit Card (Mastercard)'),
    ('Bilt', 'Bilt Mastercard (issued by Evolve Bank & Trust)'),
    ('Bilt', 'Bilt Mastercard (issued by Wells Fargo)'),
    ('Borgata Hotel Casino & Spa', 'M life Rewards Mastercard'),
    ('Caesars', 'Total Rewards Visa Credit Card'),
    ('Deserve', 'SelfScore Achieve Mastercard for Students'),
    ('Deserve', 'SelfScore Classic Mastercard'),
    ('Dell', 'Dell Preferred Account (Consumer Credit)'),
    ('Karat', 'Karat Black Card'),
    ('Mercury Financial', 'CreditShop Credit Cards'),
    ('Mission Lane', 'Mission Lane Cash Back Visa Credit Card'),
    ('Mission Lane', 'Mission Lane Visa Credit Card'),
    ('Onbe', 'Ecount'),
    ('Onbe', 'Citi Prepaid Card Services'),
    ('Onbe', 'Swift Prepaid Solutions'),
    ('Onbe', 'North Lane Technologies'),
    ('Onbe', 'daVinci Payments'),
    ('Target', 'Target RedCard Mastercard (Co-Branded)'),
    ('Target', 'Target REDcard - Credit Card (Private Label)'),
    ('Target', 'Target Guest Card (Private Label Credit Card)'),
    ('ACI Worldwide', 'ACI UP Bill Payment'),
    ('ACI Worldwide', 'BASE24'),
    ('Alacriti', 'Cosmos Payments Hub'),
    ('Alacriti', 'Cosmos for RTP'),
    ('Bold Integrated Payments', 'Priority I.S. Payment Processing'),
    ('Checkout.com', 'Opus Payments'),
    ('Circle', 'Circle Business Accounts'),
    ('DigniFi', 'Confident Financial Solutions'),
    ('Fintainium', 'ePayRails B2B Payments Platform'),
    ('Global Payments', 'OpenEdge'),
    ('Gravity Payments', 'Price & Price (Merchant Services)'),
    ('MetalPay', 'Proton Blockchain'),
    ('Mazoola', 'Virtual Piggy'),
    ('Payrix', 'Payrix Embedded Payments Platform'),
    ('Selfbook', 'SIX Travel'),
    ('Wise', 'TransferWise for Banks'),
    ('Betterment', 'Betterment for Business'),
    ('Empower Financial', 'Personal Capital Advisory Services'),
    ('Evergreen Money', 'Evergreen Money Corporation'),
    ('Lively', 'HSA Investment via TD Ameritrade'),
    ('Vanguard', 'First Index Investment Trust'),
    ('Apple', 'Apple Pay Cash'),
    ('DailyPay', 'Friday Tips'),
    ('LemFi', 'Lemonade Finance (International Money Transfer)'),
    ('Ampla', 'Gourmet Growth'),
    ('BitPay', 'Copay Wallet'),
    ('Coinbase', 'GDAX'),
    ('Coinbase', 'Coinbase Pro'),
    ('Crypto.com', 'Monaco Wallet & Card App'),
    ('Fold', 'Fold Gift Card Exchange'),
    ('Lively', 'HealtheeSavings (HSA Platform)'),
    ('Rocket', 'Title Source'),
    ('Rocket', 'Amrock'),
    ('Michigan Retailers Services', 'Retailers Fund'),
    ('Yieldstreet', 'Yieldstreet Prism Fund'),
    ('Yieldstreet', 'Yieldstreet Alternative Investment Platform'),
    ('Mayfair', 'Mayfair Cash Account (Beta)'),
    ('BillGO', 'BillHero App'),
    ('Daily Racing Form', 'DRF Bets'),
    ('Achieve', 'Freedom Debt Relief'),
    ('Bill', 'Divvy (Spend & Expense Management)'),
    ('Brex', 'Brex Premium ($49/month)'),
    ('Credit Karma', 'Credit Karma Tax'),
    ('Dave', 'Dave Overdraft Protection App'),
    ('Grid', 'Grid Tax'),
    ('HealthEquity', 'EZ Receipts'),
    ('Strike', 'Zap Lightning Wallet'),
    ('Strike', 'Olympus'),
    ('SoFi', 'SoFi Crypto (Original)'),
    ('Wirex', 'E-Coin'),
    ('Block', 'Square Capital'),
    ('CAN Capital', 'AdvanceMe Merchant Cash Advance'),
    ('CAN Capital', 'NewLogic Business Loans'),
    ('Payability', 'Instant Access (Accelerated Daily Payouts)'),
    ('PayCargo', 'PayCargo Capital'),
    ('Paysafe', 'Moneybookers Digital Wallet'),
    ('Remitly', 'BeamIt Mobile (International Money Transfer)'),
    ('Skrill', 'Moneybookers Digital Wallet'),
    ('Veem', 'Align Commerce Global Payments Platform'),
    ('Switch', 'PayClub'),
    ('Sundance Payment Solutions', 'Sundance Payment Solutions by Deluxe'),
    ('EVO Payments', 'EVO Merchant Services'),
    ('EVO Payments', 'EVO Payments International (Rebrand from EVO Merchant Services)'),
    ('EBizCharge', 'Century Business Solutions (Merchant Services)'),
    ('Ally Bank', 'Ally Bank (rebrand from GMAC Bank)'),
    ('Ally Bank', 'Ally Financial (rebrand from GMAC)'),
    ('Ally Bank', 'GMAC Bank (Direct Banking)'),
    ('Ally Bank', 'Spending Account (rebrand from Interest Checking)'),
    ('Amscot', 'Amscot MoneyCard Prepaid Mastercard'),
    ('Bankwell Direct', 'Bankwell Bank (rebrand)'),
    # Batch 2
    ('Bankwell Direct', 'Stamford First Bank'),
    ('Bankwell Direct', 'The Bank of Fairfield'),
    ('Bankwell Direct', 'The Bank of New Canaan'),
    ('Braid', 'Pool (Successor to Braid)'),
    # Batch 3
    ('Elevate Pay', 'Elevate Pay (rebrand from Bloom)'),
    # Batch 4
    ('Found', 'Found (Rebrand from Indie)'),
    ('Go2bank', 'GoBank'),
    ('Go2bank', 'Green Dot Corporation'),
    # Batch 5
    ('Inter', 'Banco Inter (Rebrand from Intermedium)'),
    ('Inter', 'Inter (Rebrand from Banco Inter)'),
    ('Inter', 'Inter&Co (Corporate Rebrand / Nasdaq Listing)'),
    ('Inter', 'Intermedium Digital Account (100% Digital Checking)'),
    ('Ivy Bank', 'Cambridge Savings Bank'),
    # Batch 6
    ('Lili', 'Lili Core (Rebrand from Lili Basic)'),
    ('Lively', 'Lively (Rebrand from HealtheeSavings)'),
    ('LemFi', 'Lemonade Finance Multi-Currency Wallets'),
    # Batch 7
    ('Maza', 'Flex Consumer (Rebrand)'),
    ('Monzo', 'Monzo (rebrand from Mondo)'),
    ('Monzo', 'Mondo Prepaid Card'),
    ('N26', 'N26 (Rebrand from Number26)'),
    ('N26', 'Number26 Current Account'),
    # Batch 9
    ('Rho', 'Rho (rebrand from Rho Business Banking)'),
    ('Roger', 'Citizens Bank of Edmond'),
    # Final batch
    ('UFB Direct', 'Axos Bank'),
    ('UFB Direct', 'Bank of Internet USA (Axos Bank)'),
    ('WEX', 'WEX Bank (Rebrand from Wright Express Financial Services)'),
    ('WEX', 'Wright Express Financial Services (WEX Bank)'),
    ('Walmart', 'ONE (rebrand from Hazel)'),
    ('Walmart', 'OnePay (rebrand from ONE)'),
    ('Zenda', 'InComm Benefits (rebrand)'),
    # OFS batch 101-115
    ('Navan', 'TripActions (Corporate Travel Management)'),
    # OFS batch 131-145
    ('Paymerang', 'Paymerang, a Corpay Brand'),
    # OFS batch 161-175
    ('Slash', 'Slash Rebrand (joinslash.com to slash.com)'),
]
for company, product in REBRANDS:
    df.loc[(df['company_name'] == company) & (df['product_name'] == product), 'is_rebrand'] = True

# Rebrands disambiguated by year (for duplicate product names with different launch years)
REBRANDS_BY_YEAR = [
    ('Rocket', 'Rocket Mortgage', '2021'),  # The 2021 company-rebrand event, not the 2015 product
]
for company, product, year in REBRANDS_BY_YEAR:
    df.loc[(df['company_name'] == company) & (df['product_name'] == product) & (df['year_launched'].astype(str) == year), 'is_rebrand'] = True

FEATURES = [
    # Batch 1
    ('Alza', 'Fee-Free ATM Access'),
    ('Amscot', 'ATMs in Every Branch'),
    ('Arival Bank', 'IFE Banking License'),
    ('Aspiration', 'Plant Your Change'),
    # Batch 2
    ('BusyKid', 'BusyKid Savings Feature'),
    ('Charlie', 'Early Social Security Access'),
    ('Chime', 'Chime Stimulus & Tax Refund Early Access'),
    ('Cleo', 'Early Direct Deposit'),
    ('Bliss', 'Bliss DV (Domestic Violence Expansion)'),
    ('Bliss', 'Transition Savings Goals'),
    ('Block', 'Cash App Direct Deposit'),
    ('CD Valet', 'CD Valet Online IRA CD Opening'),
    ('Car IQ', 'Shell Merchant Integration'),
    ('Car IQ', 'Sunoco Merchant Integration'),
    ('Car IQ', 'Kum & Go Merchant Integration'),
    ('Car IQ', 'EG America Merchant Integration'),
    ('Car IQ', 'HF Sinclair Merchant Integration'),
    ('Car IQ', 'Exxon and Mobil Merchant Integration'),
    ('Car IQ', 'Verra Mobility Toll Payment Suite'),
    ('Car IQ', 'BlackBerry In-Car Wallet'),
    # Dwolla integrations
    ('Dwolla', 'Real-Time Payments (RTP Network)'),
    ('Dwolla', 'FedNow Service Integration'),
    ('Dwolla', 'Same Day ACH'),
    # EBizCharge platform features
    ('EBizCharge', 'Agent and ISO Program'),
    ('EBizCharge', 'EBizCharge EMV API'),
    ('EBizCharge', 'Surcharging Program'),
    ('EBizCharge', 'ACH/eCheck Processing'),
    ('EBizCharge', 'Hosted Checkout / Pay Now Buttons'),
    ('EBizCharge', 'CRM Integrations'),
    ('EBizCharge', 'EMV Terminal Support'),
    ('EBizCharge', 'EBizCharge Mobile App'),
    ('EBizCharge', 'eCommerce Payment Integrations'),
    ('EBizCharge', 'Recurring Billing / Auto Pay'),
    ('EBizCharge', 'Email Pay & Payment Links'),
    ('EBizCharge', 'Customer Payment Portal'),
    ('EBizCharge', 'Virtual Terminal'),
    ('EBizCharge', 'ERP/Accounting Integrations'),
    ('EBizCharge', 'EBizCharge Connect API'),
    ('EBizCharge', 'POS Integrations'),
    ('EBizCharge', 'Infor CloudSuite Integration'),
    # Fitech Payments features
    ('Fitech Payments', 'Virtual Terminal'),
    ('Fitech Payments', 'Deluxe Merchant Services'),
    ('Fitech Payments', 'Fitech by Deluxe'),
    ('Fitech Payments', 'Financial Institution Partnership Program'),
    # Dwolla additional features
    ('Dwolla', 'Mass Pay'),
    ('Dwolla', 'Dwolla Balance (Digital Wallet)'),
    # Direct Express features
    ('Direct Express', 'Direct Express EMV Chip Cards'),
    ('Direct Express', 'Direct Express Cardless Benefit Access Service'),
    # EBizCharge core
    ('EBizCharge', 'EBizCharge Payment Gateway'),
    # EDPS features
    ('EDPS', 'Online Payment Gateway'),
    ('EDPS', 'Credit & Debit Card Processing'),
    ('EDPS', 'Check Services (Conversion & Guarantee)'),
    ('EDPS', 'Mobile/Wireless Payment Processing'),
    ('EDPS', 'Surcharge / Cash Discount Program'),
    # EPX features
    ('Electronic Payment Exchange (EPX)', 'Mobile Payment Processing'),
    ('Electronic Payment Exchange (EPX)', 'E-Commerce/MOTO Processing'),
    ('Electronic Payment Exchange (EPX)', 'Chargeback Adjudication Service'),
    ('Electronic Payment Exchange (EPX)', 'Debt Repayment Processing Program'),
    ('Electronic Payment Exchange (EPX)', 'EPXPay Semi-Integrated Terminal Application'),
    ('Electronic Payment Exchange (EPX)', 'EU Direct Merchant Acquiring'),
    ('Electronic Payment Exchange (EPX)', 'Virtual Terminal'),
    # Elite Payment Processing features
    ('Elite Payment Processing', 'E-Commerce / Card-Not-Present Solutions'),
    ('Elite Payment Processing', 'Cash Discount / Dual Pricing Program'),
    # Emerald Wallet features
    ('Emerald Wallet', 'Emerald Wallet P2P Payments App'),
    ('Emerald Wallet', 'Emerald Wallet Business Payments'),
    # EPS Financial features
    ('EPS Financial', 'Refund Disbursement Options'),
    # Expensify features
    ('Expensify', 'Personal Payments & Bill Splitting'),
    ('Expensify', 'ACH Direct Deposit Reimbursement'),
    # Finexio features
    ('Finexio', 'Virtual Card by Mail'),
    ('Finexio', 'Virtual Card Payments (Email)'),
    ('Finexio', 'Finexio Express (Enhanced ACH)'),
    # Finix features
    ('Finix', 'Finix Virtual Terminal'),
    ('Finix', 'Finix Payouts'),
    # Fintainium feature
    ('Fintainium', 'Mass Disbursements'),
    # Fitech feature
    ('Fitech Payments', 'Merchant Accounts'),
    # First Payment Services feature
    ('First Payment Services (FNB)', 'Gift Card Processing'),
    # Gelt feature
    ('Gelt', 'Gelt Direct Deposit'),
    # Heartland features
    ('Heartland Payment Solutions', 'Heartland Mobile Payments'),
    ('Heartland Payment Solutions', 'Heartland eCommerce & Online Payments'),
    # Batch 1 features
    ('360 Payment Solutions', 'Text to Pay'),
    ('Aeropay', 'Aeropay Instant Payouts'),
    ('Agile Financial Systems', 'Merchant Acquiring (ISO Services)'),
    ('Agile Financial Systems', 'APEX Gateway (Virtual Terminal & E-Commerce)'),
    ('Agile Financial Systems', 'Dual Pricing Program'),
    ('Agile Financial Systems', 'APEX Checkout & Invoicing'),
    ('AGMS', 'ACH Services'),
    ('AGMS', 'Check Services'),
    ('AGMS', 'Fee Pass-Through (Non-Traditional Processing)'),
    ('AGMS', 'Nonprofit Donation Processing'),
    ('AGMS', 'Wex Voyager Fleet Card Acceptance'),
    ('AGMS', 'Mobile Payment Processing'),
    ('Agora', 'Agora Cross-Border Payments'),
    ('Agora', 'Agora ACH Payments'),
    ('Atlantech Payments', 'Partner Programs'),
    ('Atlantech Payments', 'High-Risk / Card-Not-Present Merchant Accounts'),
    ('Atlantech Payments', 'Chargeback Mitigation'),
    ('Best Products Sales and Service', 'Free ATM Placement Program'),
    ('Best Products Sales and Service', 'ATM Processing'),
    ('Best Products Sales and Service', 'Independent ATM Deployer (IAD) Partner Program'),
    ('Branch', 'Cashless Tips'),
    ('Branch', 'Contractor Payments (1099 Payouts)'),
    ('Brightwheel', 'Brightwheel Check Deposit'),
    ('Capitol Payment Systems', 'Government Payment Solutions'),
    # Batch 1 features (round 2)
    ('Car IQ', 'Circle K Pro Digital+'),
    ('Car IQ', 'Car IQ Pay Mobile App'),
    ('Clearent', 'EMV Chip Certification'),
    ('Clearent', 'Cash Discounting / Surcharging Program'),
    ('Cardpayment Services', 'Fast Pass Funding'),
    ('Cardpayment Services', 'ZeroPay Program'),
    ('Cardpayment Services', 'Partner Solutions (ISO/Agent Program)'),
    ('Apto Payments', 'Dwolla Integration'),
    ('Earnin', 'Lightning Speed'),
    ('Clearly Payments', 'U.S. Market Expansion'),
    ('Brightwell', 'Brightwell Navigator Mobile App'),
    ('Card.com', 'Direct Deposit (QuickPay)'),
    ('EVO Payments', 'EVO Payments, Inc. (IPO & Rebrand)'),
    ('Fullpay', 'ACH and eCheck Processing'),
    # IntelliPay features
    ('IntelliPay', 'OneTerminal (Virtual Terminal)'),
    ('IntelliPay', 'eCash (Paysafecash Integration)'),
    ('IntelliPay', 'IntelliPay IVR Payment System'),
    ('IntelliPay', 'Dual Pricing / Consumer Choice Program'),
    ('IntelliPay', 'GovTeller'),
    # Lumanu features
    ('Lumanu', 'Lumanu Wallet'),
    ('Lumanu', 'Lumanu Creator Protection'),
    ('Lumanu', 'Purchasing Cards'),
    # Melio features
    ('Melio', 'Virtual Cards (Single-Use)'),
    ('Melio', 'Real-Time Payments'),
    ('Melio', 'Melio Platinum'),
    ('Melio', 'Pay by Card'),
    # Imerchant Direct features
    ('Imerchant Direct', 'Merchant Fee Free Processing (Non-Cash Charge)'),
    ('Imerchant Direct', 'Custom Rate Processing'),
    ('Imerchant Direct', 'Flat Rate Processing'),
    ('Imerchant Direct', 'E-Commerce Solutions'),
    ('Imerchant Direct', 'Check Processing (ACH)'),
    ('Imerchant Direct', 'Wireless Payment Solutions'),
    # Inchek features
    ('Inchek Merchant Services', 'High-Risk Merchant Accounts'),
    ('Inchek Merchant Services', 'Mobile Payment Processing'),
    ('Inchek Merchant Services', 'Electronic Check Re-Presentment'),
    ('Inchek Merchant Services', 'Automated Recurring Payments'),
    ('Inchek Merchant Services', 'Check Processing and Remote Deposit'),
    ('Inchek Merchant Services', 'NSF Recovery and Collections'),
    # Merchant E-Solutions features
    ('Merchant E-Solutions', 'Virtual Terminal'),
    ('Merchant E-Solutions', 'Interchange Optimization Savings Program (Level II/III)'),
    ('Merchant E-Solutions', 'Mobile Payment Processing'),
    ('Merchant E-Solutions', 'E-Commerce Payment Processing'),
    # Kasheesh mobile wallet feature
    ('Kasheesh', 'Kasheesh Mobile Wallet Dashboard (Tap-to-Pay)'),
    # Lavu / LemFi / Klarna features
    ('Lavu', 'Dual Pricing / Cash Discount Program'),
    ('LemFi', 'Request Money / Payment Links'),
    ('Klarna', 'Klarna Invoice Payments (Pay After Delivery)'),
    # Housecall Pro features
    ('Housecall Pro', 'Instapay'),
    ('Housecall Pro', 'Housecall Pro Mobile Check Deposit'),
    # Incomm Reload feature
    ('Incomm Payments', 'Vanilla Reload Network'),
    # MassPay features
    ('MassPay', 'MassPay Wallet'),
    ('MassPay', 'MassPay Instant Payout Solutions'),
    ('MassPay', 'MassPay Payout Orchestration Platform'),
    # Merchant E-Solutions features
    ('Merchant E-Solutions', 'ACH Payment Processing'),
    ('Merchant E-Solutions', 'Bank Partner / Agent Banking Programs'),
    ('Merchant E-Solutions', 'Pre-Imbursements'),
    ('Merchant E-Solutions', 'EmployeePlus Payouts'),
    # Merchant Tree features
    ('Merchant Tree Financial Services', 'Cash Discounting Program'),
    ('Merchant Tree Financial Services', 'Membership Billing'),
    ('Merchant Tree Financial Services', 'Mobile Payments'),
    ('Merchant Tree Financial Services', 'eCommerce & Shopping Cart Integration'),
    ('Merchant Tree Financial Services', 'Point-of-Sale Software'),
    ('Merchant Tree Financial Services', 'Growth Platform'),
    # Merchant World
    ('Merchant World', 'Dual Pricing Program'),
    # Michigan Retailers features
    ('Michigan Retailers Services', 'Surcharge and Cash Discount Programs'),
    ('Michigan Retailers Services', 'ACH Processing'),
    ('Michigan Retailers Services', 'Retailers Processing Network'),
    # National Cash Systems feature
    ('National Cash Systems', 'Credit Card Terminal Sales & Processing'),
    # National Credit Card Processing Group features
    ('National Credit Card Processing Group', 'MX Merchant Online Gateway'),
    ('National Credit Card Processing Group', 'NCCP Group Mobile Payments'),
    # National Processing LLC features
    ('National Processing LLC', 'Cash Discount Program'),
    ('National Processing LLC', 'ACH/eCheck Processing'),
    ('National Processing LLC', 'Credit Card Processing'),
    ('National Processing LLC', 'High-Risk Merchant Processing'),
    # NationalLink features
    ('NationalLink', 'Cannabis ATM Solutions'),
    ('NationalLink', 'NationalLink Payments (Expanded Merchant Services)'),
    # Netspend features
    ('Netspend', 'Virtual Card Numbers'),
    ('Netspend', 'Retail Reload Network'),
    # Nexs Card CapX
    ('Nexs Card', 'CapX Payments (Merchant Services)'),
    # HealthEquity entity
    ('HealthEquity', 'HealthEquity Payments, LLC'),
    # Obligo duplicate (Obligo Electronic Deposit Refund appears twice)
    # Keep first, flag second as feature
    # Actually both are feature-level vs. core product
    # North American Bancard
    ('North American Bancard', 'Check Conversion & Guarantee'),
    # Obligo duplicate
    ('Obligo', 'Obligo Electronic Deposit Refund'),
    # Monerepay duplicate
    ('Monerepay', 'MonerePay Payment Gateway'),
    # PayArc features
    ('PayArc', 'Agent & ISO Partner Program'),
    ('PayArc', 'In-Person Payment Processing'),
    ('PayArc', 'Online Payment Processing'),
    # Payment Alliance features
    ('Payment Alliance International (PAI)', 'Dynamic Currency Conversion'),
    ('Payment Alliance International (PAI)', 'Merchant Services Division'),
    # Payment Depot features
    ('Payment Depot', 'Payment Depot Virtual Terminal'),
    ('Payment Depot', 'Payment Depot Mobile Payments'),
    # PaymentClub features
    ('PaymentClub', 'Online Payment Processing'),
    ('PaymentClub', 'Surcharging and Dual Pricing'),
    ('PaymentClub', 'Cash Discount Program (ClubZero)'),
    # Paymerang feature
    ('Paymerang', 'Paymerang Positive Pay'),
    # Paymints features
    ('Paymints', 'Real-Time Payments (RTP) via Cross River'),
    ('Paymints', 'MuniPay'),
    # Payoneer features
    ('Payoneer', 'Mass Payouts'),
    ('Payoneer', 'B2B Payment Service (Request a Payment)'),
    ('Payoneer', 'Payoneer Checkout'),
    # PayCargo feature
    ('PayCargo', 'PayCargo Express (Guest Payment)'),
    # PayForward (feature, not a card)
    ('PayForward', 'PayForward Visa Link Card'),
    # Paypal QR
    ('Paypal', 'PayPal QR Code Payments'),
    # Pay.Com feature
    ('Pay.Com', 'Payment Requests / Pay Links'),
    # Happy Money
    ('Happy Money', 'Direct Card Payoff™ (Method Financial Integration)'),
    # Pcbancard features
    ('Pcbancard', 'Level 2/3 B2B Processing'),
    ('Pcbancard', 'E-Commerce & Gateway Solutions'),
    ('Pcbancard', 'Bank Referral Partnership Program'),
    ('Pcbancard', 'Cash Discount / Dual Pricing Program'),
    # Petroleum Card Services features
    ('Petroleum Card Services', 'EBT Card Acceptance'),
    ('Petroleum Card Services', 'E-Commerce and Virtual Terminal'),
    ('Petroleum Card Services', 'Wireless Payment Processing'),
    ('Petroleum Card Services', 'PCS Southern U.S. Agent Network'),
    ('Petroleum Card Services', 'Mobile Payment Solutions'),
    # Phoenix Payment Systems features
    ('Phoenix Payment Systems', 'Debit Card Processing'),
    ('Phoenix Payment Systems', 'Credit Card Processing'),
    ('Phoenix Payment Systems', 'ACH/Electronic Check Processing'),
    # Pinnacle Payments feature
    ('Pinnacle Payments', 'Cash Discount & Surcharge Programs'),
    # Paystand features
    ('Paystand', 'Branded Payment Portal'),
    ('Paystand', 'Paystand B2B Network (Bank-to-Bank)'),
    # PaySimple features
    ('PaySimple', 'Recurring Billing & Subscription Management'),
    ('PaySimple', 'Mobile Payment App'),
    ('Maza', 'Maza Mobile App'),
    ('Mazlo', 'Mazlo Mobile App'),
    ('HoneyBook', 'HoneyBook Tap to Pay'),
    ('ScanPay', 'ScanPay Tap to Pay'),
    # Pay.Com sub-products → features (keep Payment Orchestration Platform as core)
    ('Pay.Com', 'Open Banking / Pay by Bank'),
    ('Pay.Com', 'No-Code Payments'),
    ('Pay.Com', 'ACH Payments'),
    ('Pay.Com', 'Global Payment Processing'),
    ('Pay.Com', 'Pay for Agencies'),
    ('Pay.Com', 'Pay.com API & SDKs'),
    ('Pay.Com', '3D Secure 2.0 Authentication'),
    ('Pay.Com', 'Card Account Updater'),
    ('Pay.Com', 'Customizable Checkout Page'),
    ('Pay.Com', 'E-Commerce Platform Integrations'),
    ('Pay.Com', 'Customer Insights Dashboard'),
    ('Pay.Com', 'Pay.com Payment Gateway'),
    # Merchant Tree
    ('Merchant Tree Financial Services', 'Credit Card Processing'),
    # Quyana referral
    ('Quyana Card', 'QuyanaCARD Referral Program'),
    # Redstone features
    ('Redstone Payment Solutions', 'Mobile Payment Processing'),
    ('Redstone Payment Solutions', 'Check Processing Services'),
    ('Redstone Payment Solutions', 'Cash Discount Program'),
    # Revel features
    ('Revel Systems', 'Revel SmartPay'),
    ('Revel Systems', 'Revel EMV Processing'),
    # Rellevate features
    ('Rellevate', 'NGO/Global Disbursement Platform'),
    ('Rellevate', 'Government Disbursement Platform'),
    # ScanPay features
    ('ScanPay', 'ScanPay ACH Payments'),
    ('ScanPay', 'ScanPay Instant Pay'),
    ('ScanPay', 'ScanPay QR Code Payments'),
    # Scratchpay
    ('Scratchpay', 'Wellness & Subscription Plans'),
    # Safe Harbor
    ('Safe Harbor Financial', 'Expanded Payments Portfolio (Lüt & GreenCard)'),
    # Safaripay
    ('Safaripay (PaySii)', 'Mobile Airtime Top-Up'),
    # SignaPay features (keep Credit & Debit Card Processing core)
    ('SignaPay', 'Virtual Terminal'),
    ('SignaPay', 'Mobile Payment Processing'),
    ('SignaPay', 'E-Commerce Solutions'),
    ('SignaPay', 'ACH/Check Processing'),
    # Signature Payments features
    ('Signature Payments', 'Edge Cash Discount Program'),
    ('Signature Payments', 'E-Commerce & MOTO Processing'),
    ('Signature Payments', 'High-Risk Merchant Accounts'),
    ('Signature Payments', 'eCheck Processing'),
    ('Signature Payments', 'Level III Processing'),
    ('Sincere', 'Sincere Mobile App'),
    ('Slash', 'Slash Outgoing Payments (Wires & ACH)'),
    ('Splitwise', 'Splitwise Pay by Bank (via Tink)'),
    ('SpotOn', 'SpotOn Virtual Terminal'),
    ('SignaPay', 'PayLo Cash Discount'),
    ('Stax', 'Mobile Payment Processing'),
    # Sundance features
    ('Sundance Payment Solutions', 'ACH Transaction Processing'),
    ('Sundance Payment Solutions', 'Gift Cards, Check Readers & ATMs'),
    ('Sundance Payment Solutions', 'E-Commerce, Virtual Terminal & Payment Gateways'),
    ('Stripe', 'Stripe Link'),
    ('Taluspay', 'Talus Pay Advantage (Cash Discount/Surcharge Program)'),
    ('TCBPay', 'High-Risk Merchant Processing'),
    ('TCBPay', 'ACH Processing'),
    ('Teampay', 'Multi-Currency Payments'),
    ('The Bankcard Group', 'E-Commerce Solutions'),
    ('The Bankcard Group', 'Payment Terminals and Equipment'),
    ('The Bankcard Group', 'POS Software Integrations'),
    ('The Bankcard Group', 'Electronic Check Processing'),
    ('The Bankcard Group', 'Online Reporting'),
    ('The Bankcard Group', 'Gift Card and Loyalty Card Programs'),
    ('ThinkPoint Financial', 'Check Processing & EBT Services'),
    ('ThinkPoint Financial', 'EMV Terminals & NFC Contactless Payments'),
    ('TipHaus', 'hausdirect'),
    ('Take Command Health', 'AutoPay (Premium Payments)'),
    ('Toast', 'Toast Mobile Payment App'),
    ('Total Merchant Services', 'Virtual Terminal & E-Commerce Processing'),
    ('Total Merchant Services', 'Check Processing & Gift Card Programs'),
    ('Totem', 'Cash Load via Green Dot'),
    ('Truss Payments', 'Balance Protection'),
    ('Trustly', 'Instant Pay-Ins (Request for Payment)'),
    ('Trustly', 'Bill Payments Platform'),
    ('Trustly', 'Instant Payouts (RTP + FedNow)'),
    ('Tuvoli', 'Tuvoli Digital Checkout'),
    ('Tuvoli', 'Tuvoli Guaranteed Payments'),
    ('Tuvoli', 'Tuvoli Aviation Merchant Processing'),
    ('Universal Processing', 'Full-Service Provider with Own BIN'),
    ('Uber', 'Uber Wallet'),
    # Ustransact features
    ('Ustransact', 'Electronic Check Conversion'),
    ('Ustransact', 'American Express OnePoint Program'),
    ('Ustransact', 'Electronic Benefits Card (EBT) Processing'),
    ('Ustransact', 'ISO and Association Partner Programs'),
    ('Ustransact', 'Bank Partner Referral Program'),
    ('Ustransact', 'Mobile Payment Solutions'),
    ('Ustransact', 'e-Commerce Solutions'),
    ('Vecter Technologies', 'Wireless and Mobile Payment Solutions'),
    ('Vecter Technologies', 'Square Concierge by Vecter'),
    ('Venmo', 'Venmo Groups'),
    ('Venmo', 'Pay with Venmo (Merchant Payments)'),
    ('VizyPay', 'Cash Discount Program'),
    ('VizyPay', 'Dual Pricing Program'),
    ('WEX', 'Corporate Payments Solutions'),
    ('WEX', 'EV Fleet Solutions'),
    ('WEX', 'Virtual Account Numbers (VANs) for Travel'),
    ('Vecter Technologies', 'Vecter Technologies'),
    ('Veem', 'Mass Payments (MassPay)'),
    ('Wethos', 'Wethos Payments'),
    ('X1', 'X1 Virtual Cards'),
    ('Zenoti', 'Zenoti Business Payments'),
    ('Zilch', 'Zilch US Beta Launch'),
    # Zil Money platform features
    ('Zil Money', 'Check Mailing Service'),
    ('Zil Money', 'eCheck (Electronic Check) Payments'),
    ('Zil Money', 'ACH Payment Processing'),
    ('Zil Money', 'Pay Vendors by Credit Card'),
    ('Zil Money', 'Virtual Cards'),
    ('Zil Money', 'Zil Money Wallet'),
    ('Afterpay', 'Afterpay US Launch'),
    ('Affirm', 'Affirm Virtual Card'),
    ('Ally Bank', 'Health Credit Services (Ally Lending)'),
    ('American First Finance', 'CareCredit/Synchrony Integration'),
    ('Aven', 'Mortgage Refinance (Announced)'),
    ('Block', 'Afterpay (BNPL)'),
    ('Changed', 'Family & Friends (Contributors) Feature'),
    ('Changed', 'Expanded Debt Types (All Loans)'),
    ('Cherry Technologies', 'Cherry Nationwide Launch'),
    ('Cherry Technologies', 'Alle Payment Plans (powered by Cherry)'),
    ('Chime', 'Chime Early Direct Deposit'),
    ('Clair', 'Intuit QuickBooks On-Demand Pay'),
    ('College Ave', 'College Ave Student Loans'),
    ('Doc2Doc Lending', 'Student Loan Refinancing (ELFI Partnership)'),
    ('Elastic', 'Elastic Credit Reporting'),
    ('Elastic', 'Elastic Cash Advances'),
    ('Flexible Finance (Flex)', 'Federal Employee Relief Program'),
    ('EnFin', 'Residential Solar Loans (Pilot)'),
    ('Happy Money', 'Payoff, Inc. (Company Founding)'),
    ('Happy Money', 'Happy Money (Company Rebrand from Payoff, Inc.)'),
    ('Horizon Hobby', 'Klarna BNPL (Checkout)'),
    ('Horizon Hobby', 'Affirm Financing (Checkout BNPL)'),
    ('LightStream', 'Rate Beat Program'),
    ('LightStream', 'Loan Experience Guarantee'),
    ('LightStream', 'LightStream by Truist (Brand Integration)'),
    ('MoneyKey', 'Credit Access Business (Texas)'),
    ('MoneyKey', 'CC Flow Line of Credit (Texas)'),
    ('NetCredit', 'My Choice Guarantee'),
    ('Obligo', 'Reduced Deposit / Deposit-in-Installments'),
    ('Lower', 'MortgagePass'),
    ('Lower', 'Free Refi for Life'),
    ('Personify Financial', 'Personify Loan Services'),
    ('Rocket', 'Mortgage in a Box'),
    ('Sezzle', 'Sezzle Pay-in-2'),
    ('Sezzle', 'Sezzle Pay-in-5'),
    ('Sezzle', 'Sezzle Premium'),
    ('Sezzle', 'Sezzle Anywhere'),
    ('Sezzle', 'Sezzle Up'),
    ('Sezzle', 'Sezzle Monthly Plans (via partner lenders)'),
    ('Wagestream', 'Wagestream US Launch'),
    ('Wisetack', 'Progress Payments'),
    ('Wisetack', '0% APR Extended Term Add-Ons'),
    ('Ollo', 'Ollo Brand Relaunch (Merrick Bank)'),
    ('Petal', 'Petal Card, Inc. (Company)'),
    ('Sparrow Financial', 'Sparrow Rewards Program'),
    ('Checkout.com', 'Saudi mada Network Integration'),
    ('Checkout.com', 'EMI License (UK) & JCB Membership'),
    ('Checkout.com', 'American Express Global Acquiring'),
    ('Checkout.com', 'UnionPay / Diners / Discover Acquiring'),
    ('Checkout.com', 'Visa & Mastercard Acquiring Membership'),
    ('Checkout.com', 'Georgia MALPB Charter (Merchant Acquirer Limited Purpose Bank)'),
    ('Electronic Payment Exchange (EPX)', 'Rebuilt Processing Platform with End-to-End Encryption'),
    ('Gigwage', 'Gig Wage Platform Rebuild (Post-Synapse)'),
    ('Sunbit', 'Sunbit for Stripe (Stripe Integration)'),
    ('Birdie', 'Cash Registries'),
    ('BayaniPay', 'BayaniPay Prime'),
    ('Brightside', 'KickForward'),
    ('Grit Financial', 'MoneyGram Transfers'),
    ('Paypal', 'Venmo (via Braintree)'),
    ('Remitly', 'Express & Economy Delivery Tiers'),
    ('Safaripay (PaySii)', 'PaySii Mobile App (iOS/Android)'),
    ('Strike', 'El Salvador Launch'),
    ('Varo', 'Zelle Integration'),
    ('Venmo', 'Instant Transfer'),
    ('ZayZoon', 'ZayZoon Venmo Payout'),
    ('ZayZoon', 'ZayZoon Instant Gift Cards (ZayZoon Boost)'),
    ('Zil Money', 'Wire Transfer Services'),
    ('Fundbox', 'Fundbox Plus'),
    ('LendingUSA', 'PPP Loan Facilitation'),
    ('Lendio', 'Lendio PPP Loan Facilitation'),
    ('NorthOne', 'PPP Loan Facilitation'),
    ('Paintbrush', 'Same-Day Funding'),
    ('Payability', 'Payability (company founding)'),
    ('Veem', 'PPP Loan Facilitation'),
    ('Revel Systems', 'Atlas V2'),
    ('Kraken', 'Kraken Staking US Relaunch'),
    ('Kraken', 'Kraken Financial (SPDI Bank Charter)'),
    ('Grabr', 'Stablecoin (USDT/PYUSD) Integration'),
    ('Strike', 'Strike New York (BitLicense)'),
    ('Strike', 'Strike Global Launch (65+ Countries)'),
    ('Strike', 'In-House Custody & Infrastructure'),
    ('Gemini', 'Gemini Earn (Discontinued)'),
    ('Pagaya', 'Pagaya (NASDAQ: PGY) — Public Listing via SPAC'),
    ('CrowdHealth', 'Bitcoin Investment Option (Swan Bitcoin)'),
    ('Good Money', 'Impact Investment Voting'),
    ('Treasure', 'Treasure Financial'),
    ('Melio', 'Capital One Embedded AP Solution'),
    ('Melio', 'Pay Bills by Melio (Clover Integration)'),
    ('Melio', 'Shopify Bill Pay (Powered by Melio)'),
    ('Melio', 'Gusto Bill Pay & Gusto Invoicing (Powered by Melio)'),
    ('Wise', 'USD Direct Debits'),
    ('Tally', 'Late Fee Protection'),
    ('Paytrust', 'Paytrust Automatic Payment Rules'),
    # OFS mobile app entries → features of core products
    ('Build', 'Build Mobile App'),
    ('Capital on Tap', 'Mobile App'),
    ('CardConnect', 'CardPointe Mobile App'),
    ('Cashably', 'Cashably Mobile App (Consumer)'),
    ('Clearent', 'Xplor Pay Mobile App'),
    ('Cledara', 'Cledara Mobile App'),
    ('Colawallex', 'ColawalleX Mobile App'),
    ('Continental Finance', 'CFC Mobile Access App'),
    ('Credit Key', 'Credit Key Mobile App'),
    ('DigniFi', 'DigniFi Mobile App'),
    ('Fancards', 'Fancards Mobile App'),
    ('Finstro', 'MyFINSTRO Mobile App'),
    ('Float', 'Float Mobile App'),
    ('Inbanx', 'inbanx Mobile App'),
    ('Insight Card', 'Insight Mobile App'),
    ('Kasheesh', 'Kasheesh Mobile App (iOS & Android)'),
    ('Kikoff', 'Kikoff Mobile App'),
    ('Kora Money', 'Boro Mobile App'),
    ('Lively', 'Lively Mobile App'),
    ('Melio', 'Mobile App'),
    ('Mercury Financial', 'Mercury Mobile App (Digital Shopping Benefits)'),
    ('Mesh Payments', 'Mesh Mobile App'),
    ('Mission Lane', 'Mission Lane Mobile App'),
    ('Pleo', 'Spend Management Platform & Mobile App'),
    ('Quyana Card', 'Quyana Mobile App'),
    ('Saber Es Poder', 'PODERcard Digital Wallet / Mobile App'),
    ('Slash', 'Slash Mobile App'),
    ('Sunbit', 'Sunbit Mobile App'),
    ('Teampay', 'Teampaygo Mobile App'),
    ('X1', 'X1 Mobile App'),
    ('Zizu App', 'Zizu Mobile App'),
    ('Accrue Savings', 'Social Savings & Gifting'),
    ('Achieve', 'Achieve MoLO (Money Left Over)'),
    ('Achieve', 'Achieve GOOD (Get Out Of Debt)'),
    ('Airbase', 'Airbase Expense Management'),
    ('Airwallex', 'Airwallex Reimbursements'),
    ('Betterment', 'Charitable Giving'),
    ('Bonsai', 'Bonsai Expense Tracking'),
    ('Bonsai', 'Bonsai Time Tracking'),
    ('Bridge', 'Bridge Earn+ Membership'),
    ('Brigit', 'Budgeting and Financial Tracking Tools'),
    ('Capital on Tap', 'Team Management Feature'),
    ('Car IQ', 'Station Controls'),
    ('Centime', 'Expense Management (Fyle Partnership)'),
    ('CheapOAir', 'CheapOAir Save Now Buy Later (Accrue Partnership)'),
    ('Clearing', 'Clearing Owner & Accountant Portals'),
    ('Comdata', 'DRIVEN FOR COMDATA App'),
    ('Emerald Wallet', 'Emerald Wallet Budget Management'),
    ('Expensify', 'Budgeting Tool'),
    ('Expensify', 'Mileage & Distance Tracking'),
    ('Float', 'Float Reimbursements'),
    ('Inbanx', 'inbanx Automated Expense Reporting'),
    ('Uncapped', 'Sugar Powered by Uncapped (Gaming Finance)'),
    ('IntelliPay', 'Single Dip EMV (Patented Chip-Card Processing)'),
    ('North American Bancard', 'Electronic Payment Exchange (EPX)'),
    ('Paysafe', 'Netbanx Payment Processing (acquired by Neteller)'),
    ('Ally Bank', 'Fair Square Financial (Ally Credit Card)'),
    ('Zilch', 'Zilch US Beta Launch'),
    # Batch 2 features
    ('Clearent', 'Next Day Funding'),
    ('Clearent', 'Text to Pay (PayLink)'),
    ('Cliq', 'Property Management Payment Solutions'),
    ('Cliq', 'Prepaid Card Solutions'),
    ('Cliq', 'ACH Processing'),
    ('Cliq', 'E-Commerce / Remote Payment Processing'),
    ('Cliq', 'Payroll Card Services'),
    ('Clover', 'Klarna BNPL Integration'),
    ('Clover', 'Rapid Deposit'),
    ('Coast', 'Virtual Cards for Back-Office Expenses'),
    ('Cocard', 'Zero Cost Program'),
    ('Cocard', 'Billing Solutions'),
    ('Cocard', 'Check Processing'),
    ('Cocard', 'Mobile Processing Solutions'),
    ('Cocard', 'E-Commerce & Payment Gateways'),
    ('Cocard', 'EBT Processing'),
    ('Corepay', 'Corepay Healthcare Payment Processing'),
    ('Corepay', 'Corepay ACH Solutions'),
    ('Corepay', 'Corepay E-commerce Payment Processing'),
    ('Corvia', 'Specialty Acquiring (High-Risk Merchant Services)'),
    ('Corvia', 'ACH & Account-to-Account Payments'),
    ('Covercy', 'Capital Call Processing'),
    ('Datafi Payments', 'Banking and Credit Union Partner Programs'),
    ('Datafi Payments', 'Merchant Accounts'),
    ('Datafi Payments', 'Rate Lock Guarantee'),
    ('Datafi Payments', 'ISO and Agent Partner Program'),
    ('Datafi Payments', 'Mobile Payment Processing'),
    ('Defyne Payments', 'Event Payment Processing'),
    ('Defyne Payments', 'E-Commerce Payment Processing'),
    ('Defyne Payments', 'ISO / Agent Partner Program'),
    ('Defyne Payments', 'Cash Discount Program'),
    ('ConnexPay', 'Disbursements (PayOuts)'),
    ('ConnexPay', 'Classic Payment Methods (PayOuts)'),
    # Batch 3
    ('Cogni', 'Early Pay'),
    ('Credit Karma', 'Credit Karma Overdraft Coverage'),
    ('Current', 'Current Add Cash (InComm Partnership)'),
    ('Daylight', 'Daylight Savings Goals'),
    ('Earnin', 'Live Pay'),
    # Batch 4
    ('Firstcard', 'Firstcard International Student Support'),
    ('FloatMe', 'FloatMe Membership'),
    ('FloatMe', 'FloatMe Premium Membership'),
    ('Flyp Money', 'Flyp Overdraft Protection'),
    ('Fold', 'Fold+ Premium Subscription'),
    # Batch 5
    ('Insight Card', 'Direct Deposit'),
    ('Inter', 'Florida Digital Banking Branch (Fed Approval)'),
    # Batch 6
    ('Lili', 'BalanceUp (Overdraft Protection)'),
    # Batch 7
    ('Monzo', 'Monzo Perks'),
    ('Monzo', 'Pots'),
    ('Monzo', 'Salary Sorter'),
    ('Nexs Card', 'NexsCard Direct Deposit Switch'),
    ('Nexs Card', 'NexsCard Early Access to Government Benefits'),
    ('Nexs Card', 'NexsCard Mobile Deposit (Remote Deposit Capture)'),
    # Batch 8
    ('Oxygen', 'Early Direct Deposit'),
    ('Percapita', 'Penny Jar (Round-Up Savings)'),
    # Batch 9
    ('Prizepool', 'PrizePool Stacked (Paid Membership)'),
    ('Qube Money', 'Joint Accounts'),
    # Batch 10
    ('Seis', 'Seis Membership Tiers'),
    ('Spendwell', 'spendwell Overdraft Protection'),
    ('Stake Rent', 'Stake Saver (Subscription Tier)'),
    ('Sunny Day Fund', 'Goal-Based Savings & Financial Coaching'),
    ('Till Financial', 'Till Premium'),
    ('Totem', 'Early Paycheck Access'),
    ('Totem', 'Fee-Free ATM Network (Allpoint)'),
    # Final batch
    ('Walgreens', 'ATM Services'),
    ('Yotta Savings', 'Pool Play'),
    # Scraped spot-checks
    ('Card.com', 'Overdraft Protection (ODP)'),
    ('Chime', 'SpotMe (Fee-Free Overdraft)'),
    ('Community Financial Service Centers', 'ATM Services'),
    ('Greenwood', 'Greenwood Premium Membership'),
    ('Netspend', 'Optional Overdraft Service'),
    ('Porte', 'Porte Mobile Banking App'),
    ('Revolut', 'Revolut Metal Premium Subscription'),
    ('Spruce Money', 'Spruce Courtesy Coverage (Overdraft Protection)'),
    ('Venmo', 'Direct Deposit & Check Cashing'),
    ('Strike', 'Debit Card Linking & Wire Transfers'),
    # Parent/platform entries that overlap with more specific products
    ('Bm Technologies', 'BankMobile Digital Banking Platform'),
    ('H&R Block', 'H&R Block Bank'),
    ('H&R Block', 'Spruce Mobile Banking Platform'),
    ('USA National', 'USA National Digital Banking Platform'),
    ('Revolut', 'Revolut Metal (Premium Subscription)'),
    # Company founded events (not products)
    ('AppFolio', 'AppFolio (Company Founded)'),
    ('Ava', 'Ava Finance (Company Founded)'),
    ('Coast', 'Vayu (Company Founded)'),
    ('Finch', 'Finch (Company Founded)'),
    ('Fizz', 'Fizz (Company Founded)'),
    ('Globalfy', 'Globalfy (Company Founded)'),
    ('Moves Financial', 'Moves Financial (Company Founded)'),
    ('Nala', 'NALA (Company Founded)'),
    ('Novel', 'Jumpstart (Company Founded)'),
    ('Roost', 'Roost (Company Founded)'),
    ('Sequin', 'Sequin Financial Inc. (Company Founded)'),
    ('Walgreens', 'Walgreens Pharmacy & Retail (founded)'),
    ('Zil Money', 'Zil Money Corporation (Founded)'),
    # OFS batch 86-100
    ('Intuit', 'TurboTax Live'),
    ('Invoice2go', 'Invoice2go Expense Tracking'),
    ('Jeeves', 'Jeeves Expense Management'),
    ('Kikoff', 'Kikoff Subscription Cancellation'),
    ('Kleercard', 'KleerCard Expense Management Software'),
    ('Lane Health', 'Lane Health Cardmember Portal & App'),
    ('Lili', 'Real-Time Expense Tracking'),
    ('Lumanu', 'Lumanu Financial Controls'),
    # OFS batch 191-206
    ('Torpago', 'Torpago Expense Management Software'),
    ('Varo', 'Free Tax Filing'),
    ('Varo', 'Smart Bank Account Features'),
    ('Vergo', 'Credit Card Reconciliation Tool'),
    ('Vergo', 'Mobile Receipt Capture App'),
    ('Viably', 'Viably Expense Management'),
    ('Winden', 'Expense Management (Virtual Cards)'),
    ('WorkMade', 'WorkMade Expense Categorization'),
    ('Workiz', 'Expense Management'),
    ('Yield', 'Round-Up Donations'),
    # OFS batch 176-190
    ('Splitwise', 'SplitTheRent'),
    ('Splitwise', 'Splitwise Mobile App (iOS & Android)'),
    ('Spruce Money', 'Spruce Round Up Feature'),
    ('Spruce Money', 'Spruce Tax Refund Allocation'),
    ('Teampay', 'Purchase Assistant'),
    ('Teampay', 'Reimbursements'),
    ('TipHaus', 'TipHaus Employee App'),
    # OFS batch 161-175
    ('Relayfi', 'Relay Profit First Automations'),
    ('Revolut', 'Subscription Management'),
    ('Rho', 'Rho Expense Management'),
    ('Rippling', 'Rippling Spend (Expense Management)'),
    ('Rippling', 'Rippling Travel'),
    ('RiverFinance', 'Spend Wisely'),
    ('Sable', 'Sable Mobile App'),
    # OFS batch 146-160
    ('Pst', 'Crypto-Cardholder Chrome Extension'),
    ('Qapital', 'Debt Wrangler'),
    ('Qube Money', 'ProActive Budget'),
    # OFS batch 131-145
    ('Payhawk', 'Payhawk Multi-Entity Management'),
    ('Paystand', 'Smart Controls'),
    ('Prosper', 'Credit Card Optimizer'),
    # OFS batch 116-130
    ('NerdWallet', 'Cash Flow & Budget Tracking'),
    ('NerdWallet', 'Subscription & Bills Manager (via ScribeUp)'),
    ('NorthOne', 'Envelopes (Budgeting Tool)'),
    ('Onbe', 'MyPaymentVault'),
    ('PEX', 'PEX Administrative Dashboard'),
    ('PEX', 'PEX Mobile'),
    # OFS batch 101-115
    ('MaxMyInterest', 'Multi-Bank Goal Tracking'),
    ('Mazoola', 'Senior Financial Management (SFM) Product'),
    ('Mercantile', 'Mercantile Card Management Platform'),
    ('Mercury', 'Mercury Expense Management & Reimbursements'),
    ('Mesh Payments', 'Mesh Expense Management'),
    ('Mesh Payments', 'Mesh Travel Management'),
    ('Microsoft', 'Money in Excel'),
    ('Navan', 'Navan Connect (Card-Link Technology)'),
]
for company, product in FEATURES:
    df.loc[(df['company_name'] == company) & (df['product_name'] == product), 'is_feature'] = True

VARIANTS = [
    # Batch 1
    ('Atlas', 'Point Card Virtual Card'),
    # Fitech vertical variants
    ('Fitech Payments', 'Nonprofit/Church Payment Solutions'),
    ('Fitech Payments', 'Government Payment Solutions (Service Fee Model)'),
    ('Fitech Payments', 'Medical/Dental/Veterinary Payment Solutions'),
    ('WEX', 'Co-Branded and Private Label Fleet Cards'),
    # EPS Financial card variants
    ('EPS Financial', 'E1 Visa Prepaid Card'),
    ('EPS Financial', 'Gift Cards (Visa Prepaid)'),
    # Fancards variants
    ('Fancards', 'Bulk Gift Cards (B2B)'),
    ('Fancards', 'Fancards Payment Solutions'),
    # GAM Payments verticals
    ('GAM Payments', '2APay.com (Firearms Industry Processing)'),
    ('GAM Payments', 'PartnerPay.com (Multi-Location Processing)'),
    ('GAM Payments', 'FreeDonations.com (Non-Profit Processing)'),
    ('GAM Payments', 'PayCard.com (Prepaid Solutions)'),
    ('Heartland Payment Solutions', 'Heartland Campus Solutions (ECSI)'),
    # Dash Solutions verticals
    ('Dash Solutions', 'Corporate Purchasing Card Programs'),
    ('Dash Solutions', 'dashClinical'),
    ('Dash Solutions', 'Reward & Incentive Card Programs'),
    ('Dash Solutions', 'Gift Card Programs'),
    ('Dash Solutions', 'SpendIT SendIT (Disbursement Solution)'),
    # ConnexPay card variants
    ('ConnexPay', 'ConnexPay UATP Card'),
    ('ConnexPay', 'ConnexPay Flex Card'),
    ('ConnexPay', 'Global Travel Card'),
    ('QRails', 'Alight Digital Wallet (White-Label)'),
    ('Releasepay', 'MintCheetah'),
    ('Releasepay', 'RodeoPay'),
    ('Releasepay', 'CourtFunds'),
    ('Releasepay', 'ReleasePay'),
    ('Releasepay', 'RefPay'),
    ('Switch', 'Switch Clubs and Orgs'),
    # Batch 2
    ('Block', 'Cash App Card'),
    ('Block', 'Cash App Savings'),
    # Batch 3
    ('Cogni', 'Cogni Visa Debit Card'),
    ('Comun', 'Comun Visa Debit Card'),
    ('Dave', 'Dave Debit Mastercard'),
    ('Earnin', 'Earnin'),
    # Batch 4
    ('Found', 'Virtual Cards'),
    ('Greenlight', 'Greenlight Family Cash Card (for Parents)'),
    # Batch 5
    ('Instant', 'Instant Virtual Paycards'),
    ('Jassby', 'Jassby Student Travel Debit Card (WorldStrides Partnership)'),
    ('Jassby', 'Jassby Virtual Debit Card'),
    # Batch 6
    ('Lili', 'Lili Visa Debit Card'),
    ('M1', 'M1 Spend (Checking Account & Debit Card)'),
    ('Marygold & Co.', 'Marygold & Co. (UK) App'),
    ('Marygold & Co.', 'Marygold & Co. App (Beta)'),
    ('Marygold & Co.', 'Marygold & Co. App (Full Launch)'),
    # Batch 7
    ('Modak', 'MoCard (Modak Visa Debit Card)'),
    ('Monzo', 'Monzo Extra'),
    ('Monzo', 'Monzo Max'),
    ('Monzo', 'Monzo Plus'),
    ('Monzo', 'Monzo Premium'),
    ('Monzo', 'Monzo US (Full Launch)'),
    ('Monzo', 'Monzo Ireland'),
    ('Nomad', 'Nomad Visa Debit Card'),
    # Batch 8
    ('Nubank', 'Nubank Colombia'),
    ('One Finance', 'One Builder Card'),
    ('One Finance', 'One Debit Card'),
    ('Oxygen', 'Virtual Cards'),
    ('Oxygen', 'Visa Debit Card (Personal & Business)'),
    # Batch 9
    ('Qube Money', 'Visa Debit Card (Default Zero)'),
    ('Revolut', 'Revolut <18'),
    ('Revolut', 'Revolut Junior'),
    ('Revolut', 'Revolut US Launch (with Metropolitan Commercial Bank)'),
    # Batch 10
    ('Sincere', 'Sincere Rewards Debit Card (Waitlist)'),
    ('SoFi', 'Samsung Money by SoFi'),
    ('SoloFunds', 'SoLo Debit Card'),
    ('Target', 'Target Circle Debit Card'),
    ('Treecard', 'Treecard Virtual Card'),
    # Final batch
    ('Venmo', 'Venmo Mastercard Debit Card'),
    ('Yotta Savings', 'Yotta Debit Card (Mastercard/Visa)'),
    # Scraped spot-checks
    ('Current', 'Current Teen Debit Card'),
    ('DailyPay', 'Friday by DailyPay Visa Prepaid Card'),
    ('Lively', 'Lively HSA Debit Card'),
    # OFS batch 161-175
    ('Relayfi', 'Relay Pro'),
    # OFS batch 146-160
    ('Pst', 'PST Private (Teams)'),
]
for company, product in VARIANTS:
    df.loc[(df['company_name'] == company) & (df['product_name'] == product), 'is_variant'] = True

# Manual subcategory corrections
MANUAL_SUBCAT_CORRECTIONS = [
    # Batch 1
    ('Accrue Savings', 'Accrue Savings (Save Now, Buy Later Platform)', 'Other Financial Services'),
    ('Step', 'Step Black Visa Signature Card', 'Credit Cards'),
    ('7-Eleven', 'Gift Card Program', 'Loyalty/Rewards'),
    ('AtoB', 'AtoB Fuel Card', 'Credit Cards'),
    ('Comdata', 'Comdata OnRoad Card', 'Credit Cards'),
    ('7-Eleven', '7-Eleven Fleet Universal Card', 'Credit Cards'),
    ('CloudTrucks', 'CloudTrucks Fuel Network', 'Loyalty/Rewards'),
    ('Mudflap', 'Mudflap Fuel Discount App', 'Loyalty/Rewards'),
    ('TruckSmarter', 'TruckSmarter Fuel Discounts', 'Loyalty/Rewards'),
    ('Petroleum Card Services', 'Fleet Card Processing', 'Payments API'),
    # Gambling/Gaming (batch 2)
    ('Circa Las Vegas', 'Circa Sports at Tuscany', 'Gambling/Gaming'),
    ('Circa Las Vegas', 'Circa Sports', 'Gambling/Gaming'),
    # CFSC retail MSB services
    ('Community Financial Service Centers', 'Coin Services', 'Other Financial Services'),
    ('Community Financial Service Centers', 'Money Orders', 'Money Transfer'),
    ('Community Financial Service Centers', 'Check Cashing', 'Money Transfer'),
    # POS reassignments
    ('EVO Payments', 'POS Payment Processing', 'Point-of-Sale'),
    ('Fitech Payments', 'Retail POS System', 'Point-of-Sale'),
    ('Fitech Payments', 'Restaurant POS System', 'Point-of-Sale'),
    ('Fitech Payments', 'SwipeSimple POS Software', 'Point-of-Sale'),
    ('SignaPay', 'PayLo POS', 'Point-of-Sale'),
    ('Sundance Payment Solutions', 'POS Terminals & Hardware', 'Point-of-Sale'),
    # Money Transfer reassignments
    ('Expresspayments', 'Domestic Payments Service', 'Money Transfer'),
    ('Expresspayments', 'International Business Payments Platform', 'Money Transfer'),
    # Gambling/Gaming
    ('Daily Racing Form', 'DRF Bets', 'Gambling/Gaming'),
    ('Daily Racing Form', 'DRF Bets Sale / 1/ST BET PRO (Rebrand)', 'Gambling/Gaming'),
    # Consumer-issued prepaid/debit cards → Banking (Consumer)
    ('7-Eleven', 'Trans@ct by 7-Eleven Prepaid Mastercard', 'Banking (Consumer)'),
    ('7-Eleven', 'Western Union MoneyWise Prepaid Card', 'Banking (Consumer)'),
    ('7-Eleven', 'Netspend Visa Prepaid Card (at 7-Eleven)', 'Banking (Consumer)'),
    ('ADP', 'Wisely by ADP (Paycard & Digital Account)', 'Banking (Consumer)'),
    ('Blackhawk Network', 'Open-Loop Prepaid Cards', 'Banking (Consumer)'),
    ('Blackhawk Network', 'General Purpose Reloadable (GPR) Cards', 'Banking (Consumer)'),
    ('Branch', 'Branch Paycard', 'Banking (Consumer)'),
    ('Brightwell', 'OceanPay Visa Prepaid Card', 'Banking (Consumer)'),
    ('Brink\'s', 'Brink\'s Money Paycard', 'Banking (Consumer)'),
    ('Brink\'s', 'Brink\'s Money Prepaid Mastercard', 'Banking (Consumer)'),
    ('Card.com', 'Card.com Prepaid Visa Card', 'Banking (Consumer)'),
    ('Clair', 'Clair Debit Mastercard', 'Banking (Consumer)'),
    ('Comdata', 'Comdata Payroll Card', 'Banking (Consumer)'),
    ('Direct Express', 'Direct Express Debit Mastercard', 'Banking (Consumer)'),
    ('EPS Financial', 'FasterMoney Visa Prepaid Card', 'Banking (Consumer)'),
    ('Fancards', 'Fancard Prepaid Mastercard', 'Banking (Consumer)'),
    ('Global Payments', 'Netspend (Prepaid Cards)', 'Banking (Consumer)'),
    ('Godo', 'GoDo Prepaid Card', 'Banking (Consumer)'),
    ('H&R Block', 'Emerald Prepaid Mastercard', 'Banking (Consumer)'),
    # Business virtual/charge cards → Credit Cards
    ('B4B Payments', 'B4B Virtual Cards', 'Credit Cards'),
    ('Gynger', 'Gynger Virtual Card', 'Credit Cards'),
    ('GasBuddy', 'Pay with GasBuddy+ (Mastercard Charge Card)', 'Credit Cards'),
    # Gift cards → Loyalty/Rewards
    ('Blackhawk Network', 'Tap to Pay Visa Gift Card', 'Loyalty/Rewards'),
    ('Charity Charge', 'Nonprofit Gift Cards', 'Loyalty/Rewards'),
    ('Delaware North', 'Venue Cashless Prepaid Cards', 'Loyalty/Rewards'),
    ('Fancards', 'Fancard Gift Mastercard', 'Loyalty/Rewards'),
    # Batch 2 prepaid/consumer cards → Banking (Consumer)
    ('Go2bank', 'Walmart MoneyCard', 'Banking (Consumer)'),
    ('Incomm Payments', 'American Express Prepaid Card Distribution', 'Banking (Consumer)'),
    ('Incomm Payments', 'MyVanilla General Purpose Reloadable (GPR) Card', 'Banking (Consumer)'),
    ('JPay', 'paySupreme Prepaid Mastercard', 'Banking (Consumer)'),
    ('JPay', 'JPay Progress Card (Release Card)', 'Banking (Consumer)'),
    ('Kasheesh', 'Kasheesh Multi-Use Card', 'Banking (Consumer)'),
    ('Kora Money', 'KoraCard', 'Banking (Consumer)'),
    ('Kurensē', 'iPay Prepaid Mastercard', 'Banking (Consumer)'),
    ('Kurensē', 'Higher Education Disbursement Cards', 'Banking (Consumer)'),
    ('Kurensē', 'Government Benefits Disbursement Cards', 'Banking (Consumer)'),
    ('Kurensē', 'Insurance Claims Disbursement Cards', 'Banking (Consumer)'),
    ('Kurensē', 'Clinical & Market Research Incentive Cards', 'Banking (Consumer)'),
    # Business expense cards → Credit Cards
    ('Kurensē', 'Expense Reimbursement Cards', 'Credit Cards'),
    # Gift/incentive cards → Loyalty/Rewards
    ('Incomm Payments', 'Vanilla Gift Cards (Open-Loop Prepaid)', 'Loyalty/Rewards'),
    ('Kurensē', 'Employee Incentive Cards', 'Loyalty/Rewards'),
    ('Kurensē', 'Gift Cards', 'Loyalty/Rewards'),
    ('Kurensē', 'Consumer Incentive & Rebate Cards', 'Loyalty/Rewards'),
    ('Kurensē', 'Fundraising Cards', 'Loyalty/Rewards'),
    # EWA → Lending (Consumer)
    ('Kurensē', 'Earned Wage Access (EWA)', 'Lending (Consumer)'),
    # Insurance & Benefits
    ('Level', 'Level Card (Visa)', 'Insurance & Benefits'),
    # P2P → Money Transfer
    ('Marygold & Co.', 'PayAnyone', 'Money Transfer'),
    ('Mybambu', 'Bambu Pay', 'Money Transfer'),
    ('Ingo Money', 'Ingo Money App', 'Money Transfer'),
    ('MoneyGram', 'Money Orders & Official Checks (Financial Paper Products)', 'Money Transfer'),
    ('MoCaFi', 'Immediate Response Card', 'Banking (Consumer)'),
    ('MoCaFi', 'NYC Migrant Prepaid Card Program', 'Banking (Consumer)'),
    ('Netspend', 'Skylight PayOptions (Payroll Cards)', 'Banking (Consumer)'),
    ('Nexs Card', 'ExpanseFT Payroll Card Program', 'Banking (Consumer)'),
    ('Nexs Card', 'JetPay MAC Card (White-Label)', 'Banking (Consumer)'),
    ('Nexs Card', 'MPAY Payentry Card (White-Label)', 'Banking (Consumer)'),
    # Batch 2 (round 2) subcat changes
    ('Hopscotch', 'Fee-Free Instant Payments (Hopscotch Balance)', 'Money Transfer'),
    ('Novel', 'Virtual Prepaid & Credit Cards', 'Credit Cards'),
    ('OTR Clutch', 'OTR Fuel Card', 'Banking (Business)'),
    ('Obligo', 'Obligo Deposit-in-Installments', 'Lending (Consumer)'),
    ('Parallel', 'Parallel Card (Prepaid Mastercard)', 'Banking (Consumer)'),
    ('Payentry', 'Payentry Card (Prepaid Visa Payroll Card)', 'Banking (Consumer)'),
    ('Paypal', 'PayPal Debit Mastercard', 'Banking (Consumer)'),
    ('Pana', 'Pana QR Payment Link', 'Money Transfer'),
    ('Pana', 'Pana P2P Payments', 'Money Transfer'),
    ('OV Loop', 'OV Cash Card Mastercard', 'Banking (Consumer)'),
    ('Payability', 'Payability Seller Card (Visa)', 'Credit Cards'),
    ('Percapita', 'Percapita Pay (Earned Wage Access)', 'Lending (Consumer)'),
    ('PLS', 'PLS Check Cashing', 'Money Transfer'),
    ('PLS', 'Free Money Orders', 'Money Transfer'),
    ('PLS', 'Xpectations! Prepaid Mastercard', 'Banking (Consumer)'),
    ('PLS', 'Xpectations! Visa Prepaid Card', 'Banking (Consumer)'),
    ('Plata Pay', 'Plata Pay Visa Payroll Card', 'Banking (Consumer)'),
    ('Pleo', 'US Market Launch — Pathward & Evolve Bank', 'Credit Cards'),
    # QRails
    ('QRails', 'AnyDay Earned Wage Access', 'Lending (Consumer)'),
    ('QRails', 'Paycard Programs', 'Banking (Consumer)'),
    # Quyana
    ('Quyana Card', 'QuyanaCARD Prepaid Mastercard', 'Banking (Consumer)'),
    # Rellevate
    ('Rellevate', 'Rellevate Gift Card', 'Loyalty/Rewards'),
    ('Rellevate', 'Rellevate PayCard', 'Banking (Consumer)'),
    ('Skrill', 'Skrill Visa Prepaid Card (US)', 'Banking (Consumer)'),
    ('Skrill', 'Skrill Virtual Visa Prepaid Card (US)', 'Banking (Consumer)'),
    ('Slash', 'Slash Virtual Cards', 'Credit Cards'),
    ('Slash', 'Slash Reseller Card', 'Credit Cards'),
    ('Spendwell', 'spendwell Unlimited 1% Cash Back Card', 'Banking (Consumer)'),
    ('SpotOn', 'SpotOn Gift Cards', 'Loyalty/Rewards'),
    ('Stash', 'Stash Stock-Back Debit Mastercard', 'Banking (Consumer)'),
    ('Stake Rent', 'Express PayCheck', 'Lending (Consumer)'),
    ('StoreCash', 'StoreCash Mobile App', 'Loyalty/Rewards'),
    ('Topkey', 'Topkey Visa Debit & Charge Cards', 'Credit Cards'),
    ('Totem', 'P2P Payments', 'Money Transfer'),
    ('Affirm', 'Affirm Card', 'Banking (Consumer)'),
    ('Bluevine', 'Business Line of Credit (Flex Credit)', 'Lending (Business)'),
    ('Bluevine', 'Business Term Loans', 'Lending (Business)'),
    ('Bluevine', 'Invoice Factoring', 'Lending (Business)'),
    ('Oportun', 'Oportun Visa Credit Card', 'Credit Cards'),
    ('Charlie', 'Charlie Visa Debit Card', 'Banking (Consumer)'),
    ('Fizz', 'Fizz Debit Card (Mastercard)', 'Banking (Consumer)'),
    ('Fizz', 'Fizz Debit Card (Visa) / Lead Bank Partnership', 'Banking (Consumer)'),
    ('Invoice2go', 'Invoice2go Money Visa Debit Card', 'Banking (Business)'),
    ('Novo', 'Novo Mastercard Debit Card', 'Banking (Business)'),
    ('Outgo', 'Outgo Business Visa Debit Card', 'Banking (Business)'),
    ('Payoneer', 'Payoneer Prepaid Mastercard', 'Banking (Consumer)'),
    ('Poetryy Finance', 'PoetrYY Debit Card', 'Banking (Consumer)'),
    ('Porte', 'Porte Visa Debit Card', 'Banking (Consumer)'),
    ('Totem', 'Totem Debit Card', 'Banking (Consumer)'),
    ('Uncapped', 'Uncapped Business Visa Debit Card', 'Banking (Business)'),
    ('Worklyfe', 'Worklyfe Virtual Visa Debit Card', 'Banking (Consumer)'),
    ('DriveWealth', 'DriveWealth Fractional Share Trading', 'Investing'),
    ('DoorDash', 'DoorDash Drive (White-Label Logistics)', 'Fulfillment/Logistics'),
    ('Uber', 'Uber Direct', 'Fulfillment/Logistics'),
    ('Arc', 'Arc Yield Products (Money Market Funds & T-Bills)', 'Treasury Management'),
    ('Airwallex', 'Airwallex Yield (United States)', 'Treasury Management'),
    ('Crescent', 'Crescent T-Bills Access', 'Treasury Management'),
    ('Crescent', 'Crescent Brokerage Products', 'Treasury Management'),
    ('Central', 'Central Investment Advisory', 'Treasury Management'),
    ('Circle', 'Circle Invest', 'Crypto/Digital Assets'),
    ('Circle', 'Circle Trade', 'Crypto/Digital Assets'),
    ('Crypto.com', 'Crypto.com Exchange', 'Crypto/Digital Assets'),
    ('Current', 'Current Crypto Trading', 'Crypto/Digital Assets'),
    ('Gemini', 'Gemini Exchange', 'Crypto/Digital Assets'),
    ('Gemini', 'Gemini Auction (Daily Settlement)', 'Crypto/Digital Assets'),
    ('Gemini', 'Gemini Ethereum Trading', 'Crypto/Digital Assets'),
    ('Gemini', 'Gemini Foundation (Derivatives)', 'Crypto/Digital Assets'),
    ('Kraken', 'Kraken Institutional', 'Crypto/Digital Assets'),
    ('Kraken', 'Tokenized Equities (xStocks via Backed Finance)', 'Crypto/Digital Assets'),
    ('Kraken', 'Kraken OTC Desk', 'Crypto/Digital Assets'),
    ('Kraken', 'Kraken Futures (Crypto Facilities Acquisition)', 'Crypto/Digital Assets'),
    ('Kraken', 'Kraken Margin Trading', 'Crypto/Digital Assets'),
    ('Kraken', 'Kraken Dark Pool', 'Crypto/Digital Assets'),
    ('Invstr', 'Cryptocurrency Trading', 'Crypto/Digital Assets'),
    ('Rho', 'Rho Treasury', 'Treasury Management'),
    ('Treasure', 'Treasure Managed Income', 'Treasury Management'),
    ('Step', 'Step Crypto Investing (Bitcoin)', 'Crypto/Digital Assets'),
    ('Unifimoney', 'Unifimoney Crypto Trading', 'Crypto/Digital Assets'),
    ('Betterment', 'Crypto Investing by Betterment', 'Crypto/Digital Assets'),
    ('Coinbase', 'Coinbase Prediction Markets', 'Gambling/Gaming'),
    ('Crypto.com', 'OG Prediction Market', 'Gambling/Gaming'),
    ('DraftKings', 'DraftKings Predictions (Prediction Markets)', 'Gambling/Gaming'),
    ('FanDuel', 'FanDuel / CME Group Prediction Markets', 'Gambling/Gaming'),
    ('Crypto.com', 'Crypto Earn', 'Crypto/Digital Assets'),
    ('Meow', 'Meow Treasury / T-Bill Dashboard', 'Treasury Management'),
    ('Yotta Savings', 'I-Bonds Bucket', 'Investing'),
    ('Fortress Trust', 'Fortress Custody', 'Crypto/Digital Assets'),
    ('Covercy', 'Covercy Wallet (e-Fund Account)', 'Banking (Business)'),
    ('MaxMyInterest', 'Max Cash Management Platform (Individuals)', 'Banking (Consumer)'),
    ('Public', 'Public High-Yield Cash Account (HYCA)', 'Banking (Consumer)'),
    ('ACI Worldwide', 'ACI Walletron Mobile Bill Presentment', 'Bill Pay'),
    ('Betfred', 'Betfred Sports Play+ Prepaid Card', 'Gambling/Gaming'),
    ('Brex', 'Brex Travel', 'Booking/Reservations'),
    ('Brex', 'Brex Bill Pay', 'Bill Pay'),
    ('Center', 'Center Travel', 'Booking/Reservations'),
    ('DraftKings', 'DraftKings Prepaid Play+ Card', 'Gambling/Gaming'),
    ('FanDuel', 'FanDuel Prepaid Play+ Card', 'Gambling/Gaming'),
    ('Wealthfront', 'Wealthfront Cash Account', 'Banking (Consumer)'),
    ('Workiz', 'Equipment Tracking', 'Software & SaaS'),
    ('Godo', 'GoDo Earned Wage Access', 'Lending (Consumer)'),
    ('Delaware North', 'Sportservice', 'Hospitality/Dining'),
    ('Delaware North', 'Major League Sports Concessions', 'Hospitality/Dining'),
    ('Delaware North', 'Airport Food Service', 'Hospitality/Dining'),
    ('FIS', 'FIS Profile', 'Banking API'),
    ('FIS', 'Systematics Core Banking', 'Banking API'),
    ('FIS', 'FIS AffinityEdge', 'Banking API'),
    ('FIS', 'FIS Modern Banking Platform', 'Banking API'),
    ('Imerchant Direct', 'Gift and Loyalty Programs', 'Loyalty/Rewards'),
    ('Michigan Retailers Services', 'Gift Card and Loyalty Programs', 'Loyalty/Rewards'),
    ('Ustransact', 'Gift Card Program', 'Loyalty/Rewards'),
    ('Vecter Technologies', 'Gift and Loyalty Card Programs', 'Loyalty/Rewards'),
    ('Qorbis', 'Qorbis Visa Debit Card', 'Banking (Business)'),
    ('Stake Rent', 'Stake Credit Builder', 'Credit Building'),
    ('Veem', 'Veem Visa Virtual Card', 'Banking (Business)'),
    ('Venmo', 'Peer-to-Peer Payments (Venmo App)', 'Money Transfer'),
    ('Wayapay', 'Waya P2P Payments', 'Money Transfer'),
    ('WEX', 'Virtual Payments (Travel)', 'Credit Cards'),
    ('Western Union', 'USDPT Stablecoin (announced)', 'Crypto/Digital Assets'),
    ('Walgreens', 'Prepaid Card Sales (Third-Party)', 'Banking (Consumer)'),
    ('ZayZoon', 'ZayZoon Visa Prepaid Card', 'Banking (Consumer)'),
    ('Yepzy', 'Same-Day Pay', 'Lending (Consumer)'),
    ('Zytara', 'ZUSD Stablecoin', 'Crypto/Digital Assets'),
    # Gambling/Gaming
    ('Penn Entertainment', 'PENN Wallet', 'Gambling/Gaming'),
    ('Penn Entertainment', 'PENN Play Gift Cards', 'Gambling/Gaming'),
    # Other card/subcat reassignments from batch 1 re-scan
    ('Apple', 'Tap to Pay on iPhone', 'Point-of-Sale'),
    ('Bridge', 'Bridge Digital Wallet', 'Banking (Consumer)'),
    ('Benepass', 'Benepass Visa Card', 'Banking (Consumer)'),
    ('Best Friend Finance', 'Ugly Cash Visa Card', 'Banking (Consumer)'),
    ('Ambrook', 'Ambrook Visa Cards', 'Banking (Business)'),
    ('Brink\'s', 'Brink\'s Business Expense Card', 'Credit Cards'),
    ('Comdata', 'Comdata Virtual Mastercard', 'Credit Cards'),
    # Not really payments
    ('Cashably', 'Cashably Business App', 'Other Financial Services'),
    ('Cashably', 'Cashably Mobile App (Consumer)', 'Other Financial Services'),
    # P2P / money transfer moves
    ('Alviere', 'Mezu', 'Money Transfer'),
    ('Alza', 'Alza Peer-to-Peer Payments', 'Money Transfer'),
    # EWA → Lending (Consumer) per existing convention
    ('Dayforce', 'Dayforce Wallet', 'Lending (Consumer)'),
    # Crypto.com prepaid card
    ('Crypto.com', 'Crypto.com Visa Card (Rebrand from MCO Visa Card)', 'Banking (Consumer)'),
    ('B4B Payments', 'B4B Payments US Prepaid Visa Cards', 'Banking (Consumer)'),
    # Gambling/Gaming — moved from Payments
    ('1/ST', '1/ST Play+', 'Gambling/Gaming'),
    ('888', 'SI Sportsbook Play+', 'Gambling/Gaming'),
    ('Aeropay', 'Aeropay Gaming Payments', 'Gambling/Gaming'),
    ("Baldini's Casino", 'Koin Digital Wallet', 'Gambling/Gaming'),
    ("Bally's Corporation", 'Bally Bet Play+', 'Gambling/Gaming'),
    ('Betparx', 'betPARX Play+ Prepaid Card', 'Gambling/Gaming'),
    ('Betway', 'Betway Play+', 'Gambling/Gaming'),
    ('Borgata Hotel Casino & Spa', 'Borgata Sportsbook (Retail)', 'Gambling/Gaming'),
    ('Borgata Hotel Casino & Spa', 'Casino Gaming Floor', 'Gambling/Gaming'),
    ('Borgata Hotel Casino & Spa', 'Borgata Online Casino', 'Gambling/Gaming'),
    ('Boyd Gaming', 'BoydPay Digital Wallet', 'Gambling/Gaming'),
    ('Boyd Gaming', 'FanDuel Retail Sportsbooks', 'Gambling/Gaming'),
    ('Boyd Gaming', 'Stardust Online Casino', 'Gambling/Gaming'),
    ('Daily Racing Form', 'DRF Bets Sportsbook (Iowa)', 'Gambling/Gaming'),
    ('Day at the Track', 'Advance Deposit Wagering Platform', 'Gambling/Gaming'),
    ('Day at the Track', 'Pari-Mutuel Wagering Account', 'Gambling/Gaming'),
    ('Delaware North', 'Betly Online Casino', 'Gambling/Gaming'),
    ('Delaware North', 'Betly Sportsbook', 'Gambling/Gaming'),
    ('Desert Diamond Casinos', 'Bet Desert Diamond Sportsbook (Online)', 'Gambling/Gaming'),
    ('Desert Diamond Casinos', 'Bet Desert Diamond Sportsbook (Retail)', 'Gambling/Gaming'),
    ('Digital Gaming Corporation', 'Betway Play+', 'Gambling/Gaming'),
    ('Elite Casino Resorts', 'ELITE Sportsbook (Retail)', 'Gambling/Gaming'),
    ('Elite Casino Resorts', 'ELITE Sportsbook (Mobile App)', 'Gambling/Gaming'),
    ('Four Winds Casinos', 'Four Winds Online Casino & Sportsbook', 'Gambling/Gaming'),
    ('Fubo', 'Fubo Sportsbook', 'Gambling/Gaming'),
    ('Golden Nugget', 'Golden Nugget Play+', 'Gambling/Gaming'),
    ('Jackpocket', 'Jackpocket Play+', 'Gambling/Gaming'),
    ('Live! Casino Hotel', 'PlayLive! Play+ Prepaid Card', 'Gambling/Gaming'),
    ('Miami Valley Gaming', 'MVGBet/Betly Play+ Prepaid Card', 'Gambling/Gaming'),
    ('Mohegan Sun', 'MoheganSun Play+ Card', 'Gambling/Gaming'),
    ('National Cash Systems', 'Gaming Cash Access Solutions', 'Gambling/Gaming'),
    ('NYRA', 'NYRA Bets Play+', 'Gambling/Gaming'),
    ('Oaklawn', 'Parimutuel Wagering', 'Gambling/Gaming'),
    ('Oaklawn', 'Oaklawn Sports Play+ Prepaid Card', 'Gambling/Gaming'),
    ('Ocean Casino Resort', 'Ocean Play+ Prepaid Card', 'Gambling/Gaming'),
    ('Party Poker', 'Party Play+ Prepaid Card', 'Gambling/Gaming'),
    ('Parx Casino', 'Parx Play+ Prepaid Card & Parx Wallet', 'Gambling/Gaming'),
    ('Pennsylvania Lottery', 'PA Lottery Play+ Prepaid Card', 'Gambling/Gaming'),
    ('Playstar', 'PlayStar Play+ Prepaid Card', 'Gambling/Gaming'),
    ('PointsBet', 'PointsBet Prepaid Mastercard', 'Gambling/Gaming'),
    ('Soaring Eagle Casino', 'Eagle Casino & Sports (PlayEagle)', 'Gambling/Gaming'),
    ('Soaring Eagle Casino', 'Ascend Sportsbook & Lounge', 'Gambling/Gaming'),
    ('Trustly', 'Trustly Scan & Pay (Cashless Gaming)', 'Gambling/Gaming'),
    ('AngelList', 'Networked Banking (for Venture Funds)', 'Investing'),
    # Batch 2
    ('Barstool Sportsbook', 'Barstool Sportsbook & Casino Play+ Card', 'Other Financial Services'),
    ('Betfred', 'Betfred Sports Play+ Prepaid Card', 'Other Financial Services'),
    ('Betterment', '529 Education Savings Solution', 'Investing'),
    # Batch 3
    ('CreditStrong', 'FreeKick (Teen Credit Builder & Identity Protection)', 'Credit Building'),
    ('DraftKings', 'DraftKings Prepaid Play+ Card', 'Other Financial Services'),
    ('Empower Financial', '401(k) Defined Contribution Plan Recordkeeping', 'Investing'),
    ('Empower Financial', 'Section 403(b) Retirement Plans', 'Investing'),
    ('Empower Financial', 'Section 457 Deferred Compensation Plans', 'Investing'),
    ('Evergreen Money', 'Liquid Treasuries', 'Treasury Management'),
    ('FanDuel', 'FanDuel Prepaid Play+ Card', 'Other Financial Services'),
    # Batch 4
    ('Fidelity Investments', '401(k) Workplace Retirement Plans', 'Investing'),
    ('Fidelity Investments', 'Fidelity Brokerage Services (Discount Broker)', 'Investing'),
    ('Fidelity Investments', 'Fidelity Rewards Visa Signature Card', 'Credit Cards'),
    # Batch 5
    ('HealthEquity', 'Health Savings Account (HSA)', 'Insurance & Benefits'),
    ('Jetty', 'Jetty Credit', 'Lending (Consumer)'),
    ('Kraken', 'Kraken Financial (SPDI Bank Charter)', 'Crypto/Digital Assets'),
    # Batch 6
    ('LPL Financial', 'Independent Broker-Dealer Platform', 'Investing'),
    ('LPL Financial', 'LPL Institution Services', 'Investing'),
    ('Mana Pacific', 'Mana Pacific Credit History Builder', 'Credit Building'),
    ('MassMutual', '401(k) Plan Services', 'Investing'),
    ('MassMutual', 'Group Pension Trust Products', 'Investing'),
    ('MassMutual', 'MML Investors Services (Broker-Dealer)', 'Investing'),
    # Batch 7
    ('Money Network', 'Economic Impact Payment (EIP) Card', 'Other Financial Services'),
    ('Mazoola', 'Senior Financial Management (SFM) Product', 'Other Financial Services'),
    # Batch 8
    ('Oak + Fort', 'Save Now, Buy Later (via Accrue Savings)', 'Other Financial Services'),
    ('PayForward', 'PayForward Healthcare Benefits Platform (myTotal Benefits)', 'Insurance & Benefits'),
    ('Paysend', 'Paysend Credit Builder (UK)', 'Credit Building'),
    ('Percapita', 'Rent Reporting & Credit Building', 'Credit Building'),
    # Batch 9
    ('Prizepool', 'PrizePool Vault (4% APY Stablecoin-Backed)', 'Crypto/Digital Assets'),
    ('Score Media and Gaming', 'theScore Bet Play+ Prepaid Card', 'Other Financial Services'),
    # Batch 10
    ('SoloFunds', 'SoLo Credit Reporting', 'Credit Building'),
    ('Starship', 'Starship HSA Platform', 'Insurance & Benefits'),
    ('Tipico', 'Tipico Play+ Prepaid Card', 'Other Financial Services'),
    # Final batch
    ('USA National', 'USA National Visa Credit Card', 'Credit Cards'),
    ('Vanguard', 'Institutional Retirement Services (401(k) Plans)', 'Investing'),
    ('Walgreens', 'myRewards Credit Card (Synchrony)', 'Credit Cards'),
    ('Zenda', 'InComm Benefits HRA', 'Insurance & Benefits'),
    ('Zenda', 'Zenda FSA Platform', 'Insurance & Benefits'),
    # Scraped spot-checks
    ('Fidelity Investments', 'Fidelity HSA (Health Savings Account)', 'Insurance & Benefits'),
    ('Greenlight', 'Greenlight Giving', 'Other Financial Services'),
    ('Lively', 'HealtheeSavings (HSA Platform)', 'Insurance & Benefits'),
    ('Accrue Savings', 'Social Savings & Gifting', 'Other Financial Services'),
    ('CheapOAir', 'CheapOAir Save Now Buy Later (Accrue Partnership)', 'Other Financial Services'),
    # Business reclassifications
    ('Cledara', 'Cledara Virtual Cards (SaaS Payments)', 'Banking (Business)'),
    ('Comdata', 'Comdata OnRoad Card', 'Banking (Business)'),
    ('AtoB', 'AtoB Fuel Card', 'Banking (Business)'),
    ('Bonsai', 'Bonsai Card', 'Banking (Business)'),
    # Non-banking products reclassified
    ('Bridge', 'Bridge Digital Wallet', 'Payments'),
    ('Bridge', 'Bridge Earn+ Membership', 'Other Financial Services'),
    ('Braid', 'Braid Pools (Shared Wallets)', 'Other Financial Services'),
    ('Laurel Road', 'Laurel Road for Doctors', 'Banking (Business)'),
    # Debit cards misclassified as Payments -> Banking
    ('Accrue Savings', 'Accrue Debit Card', 'Banking (Consumer)'),
    ('Bankwell Direct', 'Bankwell Debit Card', 'Banking (Consumer)'),
    ('Best Friend Finance', 'Ugly Cash Mastercard Debit Card', 'Banking (Consumer)'),
    ('Bliss', 'Bliss Visa Debit Card', 'Banking (Consumer)'),
    ('Bloxley', 'Bloxley Debit Card', 'Banking (Consumer)'),
    ('Branch', 'Branch Messenger Mastercard Debit Card', 'Banking (Consumer)'),
    ('Broxel Pay', 'Miami Dolphins Prepaid Debit Card', 'Banking (Consumer)'),
    ('Coppel Access', 'Coppel Access Visa Debit Card', 'Banking (Consumer)'),
    ('Crowded', 'Crowded Visa Debit Card', 'Banking (Business)'),
    ('Ellevest', 'Ellevest Debit Card', 'Banking (Consumer)'),
    ('First Dollar', 'First Dollar Debit Card', 'Banking (Consumer)'),
    ('Flyp Money', 'Flyp Visa Debit Card', 'Banking (Consumer)'),
    ('Fold', 'Fold Visa Prepaid Debit Card', 'Banking (Consumer)'),
    ('Go2bank', 'Green Dot Prepaid Debit Cards', 'Banking (Consumer)'),
    ('Good Money', 'Good Money Debit Card', 'Banking (Consumer)'),
    ('Grabr', 'GrabrFi US Debit Card', 'Banking (Consumer)'),
    ('Guava', 'Guava Mastercard Debit Card', 'Banking (Business)'),
    ('Holdings', 'Holdings Visa Debit Card', 'Banking (Business)'),
    ('Juno', 'Juno Debit Card (Basic & Metal)', 'Banking (Consumer)'),
    ('Mana', 'Mana Visa Debit Card', 'Banking (Consumer)'),
    ('MassPay', 'MassPay Visa Debit Card Program', 'Banking (Business)'),
    ('Moves Financial', 'Moves Visa Debit Card', 'Banking (Consumer)'),
    ('Mybambu', 'MyBambu Visa Debit Card', 'Banking (Consumer)'),
    ('Netspend', 'Netspend Prepaid Visa/Mastercard Debit Card', 'Banking (Consumer)'),
    ('NorthOne', 'NorthOne Mastercard Business Debit Card', 'Banking (Business)'),
    ('Novel', 'Novel Visa Debit Card', 'Banking (Business)'),
    ('Omnimoney (Boost Mobile)', 'OmniMoney Visa Debit Card', 'Banking (Consumer)'),
    ('Pana', 'Pana Mastercard Debit Card', 'Banking (Consumer)'),
    ('Paypal', 'Venmo Debit Card', 'Banking (Consumer)'),
    ('Relayfi', 'Relay Visa Debit Cards', 'Banking (Business)'),
    ('Roger', 'ROGER Visa Debit Card', 'Banking (Consumer)'),
    ('Sacbe Payments', 'SACBE Visa Prepaid Debit Card', 'Banking (Consumer)'),
    ('Slash', 'Slash Mastercard Debit Card', 'Banking (Business)'),
    ('Stake Rent', 'Stake Visa Debit Card', 'Banking (Consumer)'),
    ('Stash', 'Stock-Back Visa Debit Card', 'Banking (Consumer)'),
    ('Switch', 'Switch Virtual Visa Debit Card', 'Banking (Consumer)'),
    ('TruckSmarter', 'TruckSmarter Visa Debit Card', 'Banking (Business)'),
    ('Unifimoney', 'Unifimoney Visa Debit Card', 'Banking (Consumer)'),
    ('Utoppia', 'Utoppia Visa Debit Card (Current)', 'Banking (Consumer)'),
    ('Utoppia', 'Utoppia Mastercard Debit Card (Original)', 'Banking (Consumer)'),
    ('Wingspan', 'Wingspan Visa Debit Card', 'Banking (Business)'),
    ('Zenda', 'Zenda Smart Debit Card', 'Banking (Consumer)'),
    ('Zeta', 'Zeta Mastercard Debit Card', 'Banking (Consumer)'),
    ('Zizu App', 'Zizu Visa Debit Card (Announced)', 'Banking (Consumer)'),
    # Exceptions from debit card batch
    ('Starship', 'Starship HSA Visa Debit Card', 'Insurance & Benefits'),
    # OFS batch 86-100
    ('International Payment Solutions', 'Virtual Terminal', 'Point-of-Sale'),
    # OFS batch 191-206
    ('Tipico', 'Tipico Play+ Prepaid Card', 'Gambling/Gaming'),
    # Corporate/charge cards in Banking
    ('Beam', 'Beam Visa Card', 'Credit Cards'),
    # HSA cards
    ('Lively', 'Lively HSA Debit Card', 'Insurance & Benefits'),
    # Miscategorized prepaid debit
    ('Block', 'Square Credit Card', 'Banking (Consumer)'),
    # Play+ stragglers
    ('Barstool Sportsbook', 'Barstool Sportsbook & Casino Play+ Card', 'Gambling/Gaming'),
    ('Betfred', 'Betfred Sports Play+ Prepaid Card', 'Gambling/Gaming'),
    ('DraftKings', 'DraftKings Prepaid Play+ Card', 'Gambling/Gaming'),
    ('FanDuel', 'FanDuel Prepaid Play+ Card', 'Gambling/Gaming'),
    # OFS → Spend Management (new subcategory)
    ('Accountable', 'Accountable Expense Capture', 'Spend Management'),
    ('Airbase', 'Airbase Spend Management Platform', 'Spend Management'),
    ('Airwallex', 'Spend Management Platform', 'Spend Management'),
    ('AtoB', 'Fleet Expense Management Dashboard', 'Spend Management'),
    ('Bill', 'BILL Spend & Expense (Rebrand from Divvy)', 'Spend Management'),
    ('Branch', 'Expense Management Cards', 'Spend Management'),
    ('Brex', 'Brex Empower (Spend Management Platform)', 'Spend Management'),
    ('Cashably', 'Cashably Business App', 'Spend Management'),
    ('Center', 'Center Expense', 'Spend Management'),
    ('Cledara', 'Cledara SaaS Management Platform', 'Spend Management'),
    ('Cledara', 'Cledara Spend (Business Expense Cards)', 'Spend Management'),
    ('Coast', 'Fleet Expense Management Platform', 'Spend Management'),
    ('Crowded', 'Crowded Expense Management', 'Spend Management'),
    ('Expensify', 'Expensify Expense Management', 'Spend Management'),
    ('Expensify', 'New Expensify (Chat-Based Super App)', 'Spend Management'),
    ('Extend', 'Universal Expense Management', 'Spend Management'),
    ('Finally', 'Finally Expense Management', 'Spend Management'),
    ('Float', 'Float Expense Management Software', 'Spend Management'),
    ('Gynger', 'Gynger Expense Dashboard', 'Spend Management'),
    ('Inbanx', 'inbanx Budget & Spend Control Platform', 'Spend Management'),
    ('Loop', 'Loop Expense Management', 'Spend Management'),
    ('Mesh Payments', 'Mesh Payments Spend Management Platform', 'Spend Management'),
    ('Payhawk', 'Payhawk Business Travel', 'Spend Management'),
    ('Payhawk', 'Payhawk Expense Management', 'Spend Management'),
    ('Qorbis', 'Qorbis Spend Management Platform', 'Spend Management'),
    ('Ramp', 'Ramp Expense Management', 'Spend Management'),
    ('Ramp', 'Ramp Plus', 'Spend Management'),
    ('Ramp', 'Ramp Travel', 'Spend Management'),
    ('Spendesk', 'Spend Management Platform', 'Spend Management'),
    ('Speedchain', 'Speedchain Expense Management Platform', 'Spend Management'),
    ('Teampay', 'Teampay Distributed Spend Management Platform', 'Spend Management'),
    ('Topkey', 'Topkey Expense Management', 'Spend Management'),
    ('Vergo', 'AI-Powered Expense Management', 'Spend Management'),
    # OFS → Tax Prep (new subcategory)
    ('Atlas Financial', 'Atlas Tax Filing Service', 'Tax Prep'),
    ('Bonsai', 'Bonsai Tax', 'Tax Prep'),
    ('Block', 'Cash App Taxes', 'Tax Prep'),
    ('Central', 'Central Accounting & Taxes', 'Tax Prep'),
    ('EPS Financial', 'EPS Financial Refund Processing Platform', 'Tax Prep'),
    ('EPS Financial', 'e-Advance Refund Transfer', 'Tax Prep'),
    ('EPS Financial', 'e-Assist (Unfunded Tax Prep)', 'Tax Prep'),
    ('EPS Financial', 'e-Bonus Refund Transfer', 'Tax Prep'),
    ('EPS Financial', 'e-Collect Refund Transfer', 'Tax Prep'),
    ('Every', 'Every for Taxes', 'Tax Prep'),
    ('Found', 'Tax Estimation & Auto-Save', 'Tax Prep'),
    ('Intuit', 'Lacerte Tax Software', 'Tax Prep'),
    ('Intuit', 'ProSeries Tax Software', 'Tax Prep'),
    ('Intuit', 'TurboTax', 'Tax Prep'),
    ('Kick', 'Tax Deduction Finder', 'Tax Prep'),
    ('Percapita', 'Tax Filer', 'Tax Prep'),
    ('Percapita', 'Tax Planner', 'Tax Prep'),
    ('PLS', 'PLS Tax Service', 'Tax Prep'),
    ('Ruby Money', 'Ruby Money Automated Tax Payments', 'Tax Prep'),
    ('Tribevest', 'Tribevest K-1 Tax Services', 'Tax Prep'),
    ('Fundomate', 'ERC Filing Program', 'Tax Prep'),
    # OFS → Loyalty/Rewards
    ('GasBuddy', 'GasBack (Card-Linked Offers Program)', 'Loyalty/Rewards'),
    # AP automation → Bill Pay
    ('Paymerang', 'KwikPayables', 'Bill Pay'),
    ('Paymerang', 'Paymerang Invoice Automation', 'Bill Pay'),
    ('Paymerang', 'Paymerang PO Automation', 'Bill Pay'),
    ('Paymerang', 'Paymerang Payment Automation', 'Bill Pay'),
    ('Paymerang', 'Paymerang Receivables Automation', 'Bill Pay'),
    ('Paymerang', 'Paymerang Vendor Enrollment & Management', 'Bill Pay'),
    ('PayCargo', 'AP Automation', 'Bill Pay'),
    ('Paystand', 'Accounts Payable Automation (via Teampay)', 'Bill Pay'),
    ('Teampay', 'AP Automation (Invoice Management)', 'Bill Pay'),
    ('Edenred Pay', 'Edenred Pay Invoice Automation', 'Bill Pay'),
    ('Payhawk', 'Payhawk Procure-to-Pay', 'Bill Pay'),
    ('Workiz', 'Workiz Card', 'Credit Cards'),
    # OFS batch 161-175
    ('Score Media and Gaming', 'theScore Bet Play+ Prepaid Card', 'Gambling/Gaming'),
    # OFS batch 146-160
    ('Ramp', 'Ramp Bill Pay (Accounts Payable)', 'Bill Pay'),
    # OFS batch 101-115
    ('Mercury', 'Mercury Bill Pay', 'Bill Pay'),
    ('Money Network', 'Economic Impact Payment (EIP) Card', 'Consumer Prepaid'),
]
for company, product, new_subcat in MANUAL_SUBCAT_CORRECTIONS:
    mask = (df['company_name'] == company) & (df['product_name'] == product)
    df.loc[mask, 'product_subcategory'] = new_subcat

# Products/companies to exclude from analysis (e.g. non-US with non-comparable rates)
EXCLUDED_COMPANIES = [
    'Broxel Pay',
]
for company in EXCLUDED_COMPANIES:
    df.loc[df['company_name'] == company, 'is_excluded'] = True

# Specific non-US products (country-specific variants of otherwise US companies)
NON_US_PRODUCTS = [
    ('Coinbase', 'Coinbase Card (International)'),
    ('Klarna', 'Klarna Savings Accounts (Europe)'),
    ('LemFi', 'Instant Access Savings Account (UK)'),
    ('LemFi', 'Global Accounts (Nigeria)'),
    ('Marcus', 'Marcus UK Online Savings'),
    ('Monzo', 'Monzo Current Account (UK)'),
    ('Monzo', 'Monzo Savings (UK - Interest-Bearing Pots)'),
    ('Monzo', 'Monzo Business Accounts'),
    ('Payfare', 'Uber Pro Card (Canada)'),
    ('N26', 'N26 Instant Savings'),
    ('N26', 'N26 Black (Premium Account)'),
    ('N26', 'N26 Joint Accounts'),
    ('N26', 'N26 Business Account'),
    ('Nubank', 'NuConta (Digital Savings Account)'),
    ('Nubank', 'NuConta PJ (Business Accounts)'),
    ('Revolut', 'Multi-Currency Prepaid Debit Card & App'),
    ('Revolut', 'Revolut Pro (Freelancer Banking)'),
    ('Wirex', 'Wirex X-Accounts (Savings)'),
    ('Wirex', 'Wirex Business'),
    ('Wise', 'Wise Debit Card (Multi-Currency Mastercard)'),
    ('Wise', 'Wise Account (Multi-Currency Account)'),
    ('Wise', 'TransferWise Debit Card (Multi-Currency Mastercard)'),
    ('Wise', 'Wise Business'),
    ('Equals Money', 'FairFX Linked Cards (B2C)'),
    ('Loop', 'Loop Multi-Currency Accounts'),
    ('Loop', 'Loop'),
    ('Changera', 'Virtual Domiciliary Bank Accounts'),
    ('Nomad', 'Nomad International Account (USD)'),
    ('Elevate Pay', 'Bloom High-Yield Savings Account'),
    ('Capital on Tap', 'Business Instant Savings Account'),
    ('Arival Bank', 'EUR SEPA Checking Account'),
    ('Banpay', 'Banpay Multi-Currency Business Account'),
    ('Float', 'Float Business Accounts'),
    ('Ibanera', 'Multi-Currency IBAN Deposit Accounts'),
    ('Colawallex', 'ColawalleX B2B Global Business Accounts'),
    ('MoneyKey', 'Propel Bank (Puerto Rico IFE)'),
    ('Payhawk', 'Payhawk Business Accounts (IBANs)'),
    ('Pleo', 'Multi-Currency Accounts'),
    ('SumUp', 'SumUp Business Account'),
    ('Changera', 'Multi-Currency Wallets'),
    ('Colawallex', 'ColawalleXPay Payment Gateway'),
    ('Colawallex', 'ColawalleX International Logistics'),
    ('Colawallex', 'ColawalleX Escrow Transactions'),
    ('Colawallex', 'ColawalleX Cross-border Platform Collection'),
    ('Colawallex', 'ColawalleX B2B Foreign Trade Collection'),
    ('Colawallex', 'ColawalleX Global Acquiring'),
    ('Colawallex', 'ColawalleX Value-Added Services'),
    ('Colawallex', 'ColawalleX Global Collection & Disbursement'),
    ('Colawallex', 'ColawalleX Local Wallet'),
    ('Equals Money', 'FairFX Currency Cards'),
    ('Equals Money', 'Equals Money (B2B Platform)'),
    ('Equals Money', 'Equals Money Expense Management'),
    ('Equals Money', 'FairFX Travel Cash'),
    ('Equals Money', 'FairFX International Payments'),
    ('Equals Money', 'Oonex SA (Acquisition)'),
    ('Equals Money', 'HermexFX (Acquisition)'),
    ('Equals Money', 'Spectrum Financial Group / CardOneMoney (Acquisition)'),
    ('Equals Money', 'Equals Money Solutions (Enterprise)'),
    ('Equals Money', 'Casco Financial Services / Equals Connect (Acquisition)'),
    ('Global Payments', 'Ezidebit'),
    ('Global Payments', 'eWay (Australia/NZ Online Payments)'),
    ('Clearly Payments', 'Recurring Payments and Subscription Billing'),
    ('Clearly Payments', 'Bank and Credit Union Partner Programs'),
    ('Clearly Payments', 'Membership Pricing Model'),
    ('Clearly Payments', 'Mobile Payment Processing'),
    ('Clearly Payments', 'Merchant Accounts'),
    ('CurrencyFair', 'Buy-World Marketplace Payment Product'),
    ('CurrencyFair', 'CurrencyFair Business'),
    ('Brightwell', 'OceanPay Prepaid Mastercard (EMV Euro)'),
    ('B4B Payments', 'Bread4Scrap'),
    ('B4B Payments', 'Payment Card Solutions Prepaid Expense Cards'),
    ('B4B Payments', 'PCS Incentives & Rewards Cards'),
    ('B4B Payments', 'PCS Payroll Cards'),
    ('Kyshi', 'Kyshi for Business (Merchant of Record)'),
    ('Loop', 'Loop Contractor Payments'),
    ('Paysafe', 'Net+ Prepaid Mastercard'),
    ('Paysafe', 'PaysafeCash'),
    ('Paysafe', 'PaysafeCard Prepaid Vouchers'),
    ('Payfacto', 'PayFacto'),
    ('Pleo', 'Smart Company Cards — Prepaid Mastercard'),
    ('Spendesk', 'Corporate Cards — Physical & Virtual'),
    ('Trustly', 'Pay N Play'),
    ('Zilch', 'Zilch Credit Reporting (Experian/TransUnion)'),
    ('Zilch', 'Zilch Pay Over 3 Months'),
    ('Zilch', 'Zilch Physical Visa Card'),
    ('Zilch', 'Zilch Pay'),
    ('Zilch', 'Zilch Pay-in-4'),
    ('Zilch', 'Zilch Virtual Mastercard'),
    ('Zilch', 'Zilch Up'),
    ('ZayZoon', 'ZayZoon Canada Re-launch'),
    ('Airwallex', 'Borderless Card (Hong Kong)'),
    ('Airwallex', 'Borderless Card (Australia)'),
    ('Banpay', 'Banpay Virtual Cards'),
    ('Capital on Tap', 'Capital on Tap Business Credit Card (UK)'),
    ('Capital on Tap', 'Preloading'),
    ('Capital on Tap', 'Capital on Tap Pro Card'),
    ('Changera', 'EasyDollar Virtual & Physical Card'),
    ('Colawallex', 'ColawalleX VCC (Visa Virtual Card)'),
    ('Finstro', 'Finstro Mastercard (Australia)'),
    ('Elevate Pay', 'Bloom Visa Card'),
    ('Float', 'Float Corporate Card (CAD)'),
    ('Float', 'Float USD Corporate Card'),
    ('Float', 'Float Virtual Cards'),
    ('Float', 'Float Cards 2.0'),
    ('Float', 'Float Charge Card'),
    ('Nubank', 'Nubank Credit Card (Mastercard)'),
    ('Nubank', 'Nu México (International Expansion)'),
    ('Nubank', 'Nubank Personal Loans'),
    ('Nubank', 'Nubank Colombia'),
    ('Nubank', 'Nubank Rewards Program'),
    ('Nubank', 'Nubank Crypto'),
    ('Nubank', 'NuInvest (Investment Platform)'),
    ('Nubank', 'Nubank Life Insurance'),
    ('Nubank', 'Nubank Travel eSIM'),
    ('Nubank', 'Nubank AI Virtual Assistant (GPT-4)'),
    ('Ramp', 'Ramp International (UK & EU Cards)'),
    ('Ramp', 'Ramp Stablecoin-Backed Corporate Cards'),
    ('Wagestream', 'Wagestream Credit Card'),
    ('WEX', 'Motorpass/Motorcharge Fleet Cards (Australia)'),
    ('WEX', 'ExxonMobil Esso Card Program (Europe)'),
    ('Wirebarley', 'WireBarley Global Card'),
    ('CurrencyFair', 'Zai Enterprise Payment Platform'),
    ('Nala', 'Rafiki B2B Payments API'),
    ('Skrill', 'Moneybookers Merchant Payment Gateway'),
    ('Wirex', 'Wirex BaaS'),
    ('Wirex', 'Wirex Agents'),
    ('Airwallex', 'Airwallex Yield (Australia)'),
    ('Marygold & Co.', 'Marygold & Co Limited (UK Advisory)'),
    ('Monzo', 'Monzo Investments'),
    ('Monzo', 'Monzo Pensions'),
    ('N26', 'N26 Stocks & ETFs (N26 Invest)'),
    ('N26', 'N26 Crypto'),
    ('Nomad', 'Nomad Investment Platform'),
    ('Dream Payments', 'Interac e-Transfer Payouts'),
    ('Nala', 'NALA EU Expansion'),
    ('Nala', 'NALA Mobile Money App (Tanzania)'),
    ('Monzo', 'Monzo-to-Monzo Instant Transfers'),
    ('Nium', 'Instarem (Consumer Remittance)'),
    ('Paysafe', 'Neteller Digital Wallet'),
    ('Paysend', 'Paysend Credit Builder (UK)'),
    ('Finstro', 'Finstro (FC Capital Cashflow Platform)'),
    ('Finstro', 'Finstro Trade Account (Australia)'),
    ('Loop', 'Loop Capital'),
    ('Loop', 'Lending Loop'),
    ('Unlimited Remit', 'Qatar Remittance Service'),
    ('Unlimited Remit', 'UAE Remittance Service'),
    ('Unlimited Remit', 'Canada to Nepal/India Remittance Service'),
    ('Unlimited Remit', 'UK to Nepal/India Remittance Service'),
    ('Unlimited Remit', 'Europe to Nepal/India Remittance Service'),
    ('Wirebarley', 'WireBarley Remittance Service'),
    ('Wirebarley', 'WireBarley B2B Remittance Platform'),
    ('Wirebarley', 'WireBarley e-Wallet'),
    ('Coinbase', 'Coinbase International Exchange'),
    ('Circle', 'EURC (Euro Coin)'),
    ('Strike', 'Strike Europe'),
    ('Wirex', 'Wirex'),
    ('Wirex', 'E-Coin Crypto-Enabled Debit Card'),
    ('Wirex', 'Wirex Token (WXT)'),
    ('Wirex', 'Cryptoback Rewards'),
    ('Wirex', 'Wirex Contactless Debit Card'),
    ('Monzo', 'Contents Insurance'),
    ('Monzo', 'Home Insurance'),
    ('N26', 'N26 Insurance'),
    ('Republic', 'Republic Europe (Seedrs)'),
    ('Delaware North', 'International Venue Services'),
    ("Payfacto", "Maitre'D POS"),
    ("Payfacto", "VelPAY"),
    ("Payfacto", "SecureTablePay"),
    ('Afterpay', 'Clearpay (UK Brand)'),
    ('Avant', 'International Expansion (UK and Canada)'),
    ('Enova', 'QuickQuid (UK)'),
    ('Enova', 'Simplic (Brazil)'),
    ('Inter', 'Inter (Rebrand from Banco Inter)'),
    ('Inter', 'Banco Inter (Rebrand from Intermedium)'),
    ('Inter', 'Inter Shop (Marketplace / Cashback)'),
    ('Inter', 'Intermedium Digital Account (100% Digital Checking)'),
    ('Inter', 'Inter DTVM (Securities Distributor / Investment Platform)'),
    ('Inter', 'Inter LOOP Rewards Program'),
    ('Inter', 'Inter Seguros (Insurance Brokerage)'),
    ('Inter', 'Real Estate Credit'),
    ('Inter', 'Intermedium Financeira'),
    ('Inter', 'Inter&Co (Corporate Rebrand / Nasdaq Listing)'),
    ('LemFi', 'Send Now, Pay Later (SNPL)'),
    ('LemFi', 'LemFi Credit'),
    ('MoneyKey', 'Fora Credit'),
    ('Monzo', 'Monzo Personal Loans (UK)'),
    ('Monzo', 'Monzo Overdrafts'),
    ('N26', 'N26 Personal Loans'),
    ('N26', 'N26 Overdraft'),
    ('Moves Financial', 'Gig Worker Personal Loans (Canada)'),
    ('Revolut', 'Consumer Lending (Personal Loans)'),
    ('Sezzle', 'Sezzle Canada Launch'),
    # OFS batch 146-160
    ('Qapital', 'Qapital Savings App (Sweden)'),
    # Depository batch 4 — FCY/LCY = foreign/local currency (non-US fintech)
    ('Unlimited Remit', 'FCY Term Deposit'),
    ('Unlimited Remit', 'LCY Term Deposit'),
    # Depository batch 6 — Mexican fintech
    ('Broxel Pay', 'Broxel Family Plan (Plan Familiar)'),
    # Depository batch 7 — UK
    ('Monzo', 'Monzo Savings (UK - Interest-Bearing Pots)'),
    # Financial (Lending Consumer) batch 1 — UK
    ('Afterpay', 'Clearpay (UK Brand)'),
    # Financial (Credit Cards) batch 1 — UK
    ('Capital on Tap', 'Capital on Tap Business Credit Card (UK)'),
    # Financial (small subcats) — non-US
    ('Broxel Pay', 'Virtual Vault (Boveda Virtual)'),  # Mexico
    ('Float', 'Float Yield'),  # Canada (CAD)
]
for company, product in NON_US_PRODUCTS:
    df.loc[(df['company_name'] == company) & (df['product_name'] == product), 'is_non_us'] = True

n_rebrands = df['is_rebrand'].sum()
n_features = df['is_feature'].sum()
n_variants = df['is_variant'].sum()
n_excluded = df['is_excluded'].sum()
n_non_us = df['is_non_us'].sum()
print(f"Flagged: {n_rebrands} rebrands, {n_features} features, {n_variants} variants, {n_excluded} excluded, {n_non_us} non-US")

# ── Cross-check subcategories against v2 extracted product types ─────────────
# When the LLM-extracted product type from scraped pages clearly conflicts
# with the master subcategory, override it.
V2_FILE = os.path.join(DATA_CLEANED_DIR, 'extracted_product_info_v2.csv')
if os.path.exists(V2_FILE):
    import csv as csv_mod
    import io
    import sys
    csv_mod.field_size_limit(sys.maxsize)

    with open(V2_FILE, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read().replace('\x00', '')
    v2_rows = list(csv_mod.DictReader(io.StringIO(text)))

    # Build lookup: (company, normalized_product) -> extracted product_type
    # Normalize names to handle parentheses/special chars stripped during scraping
    def _normalize_name(name):
        n = re.sub(r'[^a-zA-Z0-9 ]', ' ', str(name).lower())
        return re.sub(r'\s+', ' ', n).strip()

    v2_types = {}
    for r in v2_rows:
        if r.get('status') == 'OK' and r.get('product_type', '').strip():
            key = (r['company'].strip().lower(), _normalize_name(r['product_name']))
            v2_types[key] = r['product_type'].strip().lower()

    # Map extracted types to correct subcategories
    # Only map types where the correct subcategory is unambiguous
    TYPE_TO_SUBCAT = {
        'credit card': 'Credit Cards',
        'personal loan': 'Lending (Consumer)',
        'cash advance': 'Lending (Consumer)',
        'payment app': 'Payments',
        'payment gateway': 'Payments',
        'bill pay service': 'Bill Pay',
        'investment account': 'Investing',
        'crypto wallet': 'Crypto/Digital Assets',
        'credit reporting service': 'Credit Building',
        'payroll service': 'Other Financial Services',
    }

    # For debit cards classified as Credit Cards, correct to Banking
    DEBIT_TO_BANKING = {
        'debit card', 'prepaid card', 'prepaid debit account',
        'reloadable card account',
    }

    corrections = 0
    for idx, row in df.iterrows():
        key = (str(row['company_name']).strip().lower(), _normalize_name(row['product_name']))
        if key not in v2_types:
            continue
        ext_type = v2_types[key]
        master_subcat = row['product_subcategory']

        # Check if extracted type implies a different subcategory
        if ext_type in TYPE_TO_SUBCAT:
            implied = TYPE_TO_SUBCAT[ext_type]
            # Only override if the master subcategory is clearly wrong
            # (i.e. the implied subcategory is in a different tier)
            if master_subcat in ('Banking (Consumer)', 'Banking (Business)') and implied not in ('Banking (Consumer)', 'Banking (Business)'):
                df.at[idx, 'product_subcategory'] = implied
                corrections += 1
            elif master_subcat == 'Credit Cards' and implied in ('Lending (Consumer)',):
                df.at[idx, 'product_subcategory'] = implied
                corrections += 1

        # Debit cards classified as Credit Cards -> correct to Banking
        if ext_type in DEBIT_TO_BANKING and master_subcat == 'Credit Cards':
            # Determine consumer vs business from product name
            name_lower = row['product_name'].lower()
            if any(w in name_lower for w in ('business', 'corporate', 'commercial')):
                df.at[idx, 'product_subcategory'] = 'Banking (Business)'
            else:
                df.at[idx, 'product_subcategory'] = 'Banking (Consumer)'
            corrections += 1

    print(f"Cross-check with v2 extraction: {corrections} subcategory corrections applied")
else:
    print(f"No v2 extraction file found at {V2_FILE}, skipping cross-check")

# ── Parse rebrand predecessor→successor relationships ────────────────────────
# Build a canonical product name per product by following rebrand chains
# to their terminal (latest) name.

import re as _re

# Pattern 1: "X (rebrand from Y)" or "X (rebranded from Y)"
# The row IS X, predecessor name is Y, but predecessor is a separate row.
pat_from = _re.compile(r'\(rebrand(?:ed)?\s+from\s+([^)]+)\)', _re.IGNORECASE)
# Pattern 2: "X (rebranded to Y)" — the row IS X, successor is Y (another row).
pat_to = _re.compile(r'\(rebrand(?:ed)?\s+to\s+([^)]+)\)', _re.IGNORECASE)

# Build predecessor -> successor map per company
# Key: (company, predecessor_name_normalized) -> successor_name (canonical)
def _norm_name(s):
    s = _re.sub(r'\(.*?\)', '', str(s)).strip()
    s = _re.sub(r'\s+', ' ', s).lower()
    return s

# Build a lookup of all product names per company for matching
company_products = {}
for _, r in df[df['entry_type'] == 'Product'].iterrows():
    co = r['company_name']
    company_products.setdefault(co, []).append({
        'name': r['product_name'],
        'norm': _norm_name(r['product_name']),
        'year': r['year_launched'],
    })

# For each product with "rebrand from X" or "rebranded to X", find the other row
# and create a predecessor → successor mapping
rebrand_pairs = []  # list of (company, predecessor_name, successor_name)

for idx, r in df[df['entry_type'] == 'Product'].iterrows():
    name = str(r['product_name'])
    co = r['company_name']
    m_from = pat_from.search(name)
    m_to = pat_to.search(name)

    if m_from:
        # This row is the successor; find the predecessor by the name captured
        pred_hint = _norm_name(m_from.group(1))
        if not pred_hint:
            continue
        # Match against other products at this company — prefer exact or
        # the row that starts with the hint (e.g. "AvantCredit (Personal Loans)"
        # matches hint "avantcredit")
        best_match = None
        for p in company_products.get(co, []):
            if p['name'] == name:
                continue
            if p['norm'] == pred_hint or p['norm'].startswith(pred_hint + ' ') or p['norm'].split(' ')[0] == pred_hint.split(' ')[0] and pred_hint in p['norm']:
                best_match = p['name']
                break
        if best_match:
            rebrand_pairs.append((co, best_match, name))
    if m_to:
        # This row is the predecessor; find the successor
        succ_hint = _norm_name(m_to.group(1))
        if not succ_hint:
            continue
        best_match = None
        for p in company_products.get(co, []):
            if p['name'] == name:
                continue
            if p['norm'] == succ_hint or p['norm'].startswith(succ_hint + ' ') or p['norm'].split(' ')[0] == succ_hint.split(' ')[0] and succ_hint in p['norm']:
                best_match = p['name']
                break
        if best_match:
            rebrand_pairs.append((co, name, best_match))

# ── Parse rebrand pairs from descriptions ────────────────────────────────────
# Pattern: "Formerly known as X" or "Formerly called X" etc.
pat_formerly = _re.compile(
    r'[Ff]ormerly (?:known as|called|named) ([^.(,;]+?)(?:\.|\(|,|;|\s+Rebranded|\s+in\s+\d)',
    _re.IGNORECASE)

desc_pairs = []
for _, r in df[df['entry_type'] == 'Product'].iterrows():
    desc = str(r.get('product_history_and_description', ''))
    if not desc or desc == 'nan':
        continue
    co = r['company_name']
    current_name = r['product_name']
    for m in pat_formerly.finditer(desc):
        former = m.group(1).strip().rstrip('.')
        if len(former) < 3 or len(former) > 80:
            continue
        # Find matching product row in same company
        former_norm = _norm_name(former)
        for p in company_products.get(co, []):
            if p['name'] == current_name:
                continue
            # Match if the former text equals the candidate's normalized name
            # (stricter than before — exact match or starts-with)
            if p['norm'] == former_norm or p['norm'].startswith(former_norm + ' '):
                desc_pairs.append((co, p['name'], current_name))
                break

print(f"Parsed {len(desc_pairs)} additional pairs from descriptions")
rebrand_pairs.extend(desc_pairs)

# Manual overrides: blocked pairs (auto-parsing got them wrong)
BLOCKED_REBRAND_PAIRS = [
    ('Ally Bank', 'GMAC Mortgage', 'Ally Financial (rebrand from GMAC)'),
    ('Yieldstreet', 'Yieldstreet Alternative Investment Platform', 'Willow Wealth (rebrand from Yieldstreet)'),
    ('Monarch Casino', 'Clarion Hotel Casino (Reno, NV)', 'Atlantis Casino Resort (Rebrand from Clarion)'),
    ('Monarch Casino', 'Monarch Casino Black Hawk (Colorado)', 'Monarch Casino Resort Spa (Rebrand from Monarch Casino Black Hawk)'),
    ('Payment Alliance International (PAI)', 'PAI Reports', 'AMP+ (rebrand from PAI Reports)'),
    ('Changed', 'ChangEd Round-Up App', 'Expanded Debt Types (All Loans)'),
    ('Zenda', 'Zenda HSA Platform', 'InComm Benefits HRA'),
]
blocked_set = set(BLOCKED_REBRAND_PAIRS)
rebrand_pairs = [p for p in rebrand_pairs if p not in blocked_set]
print(f"Parsed {len(rebrand_pairs)} predecessor→successor pairs from names ({len(blocked_set)} blocked)")

# Follow chains to terminal name (canonical)
# Each product maps to the end of its rebrand chain
pred_to_succ = {}  # (company, pred_name) -> succ_name
for co, pred, succ in rebrand_pairs:
    pred_to_succ[(co, pred)] = succ

def canonical_name(company, name, seen=None):
    """Follow the rebrand chain until reaching the terminal name."""
    if seen is None:
        seen = set()
    key = (company, name)
    if key in seen:
        return name  # cycle protection
    seen.add(key)
    if key in pred_to_succ:
        return canonical_name(company, pred_to_succ[key], seen)
    return name

df['canonical_name'] = df.apply(
    lambda r: canonical_name(r['company_name'], r['product_name'])
    if r['entry_type'] == 'Product' else r['product_name'], axis=1)

n_redirected = (df['canonical_name'] != df['product_name']).sum()
print(f"{n_redirected} products redirected to their canonical (latest) name")

# ── Parse discontinuation years from descriptions ────────────────────────────
# Look for patterns like "discontinued in 2024" near keywords and extract years
_disc_patterns = [
    _re.compile(r'\bdiscontinued\b[^.]{0,80}?(\b(?:19|20)\d{2}\b)', _re.IGNORECASE),
    _re.compile(r'\bsunset(?:ted|ting)?\b[^.]{0,80}?(\b(?:19|20)\d{2}\b)', _re.IGNORECASE),
    _re.compile(r'\bshut\s*down\b[^.]{0,80}?(\b(?:19|20)\d{2}\b)', _re.IGNORECASE),
    _re.compile(r'\bwound\s*down\b[^.]{0,80}?(\b(?:19|20)\d{2}\b)', _re.IGNORECASE),
    _re.compile(r'\bceased\b[^.]{0,80}?(\b(?:19|20)\d{2}\b)', _re.IGNORECASE),
    _re.compile(r'\bwinds?\s*down[^.]{0,80}?(\b(?:19|20)\d{2}\b)', _re.IGNORECASE),
]

# Manual overrides: products where auto-detection gives wrong discontinuation year
# (e.g. description references a PREVIOUS product's shutdown, not this one)
DISC_YEAR_OVERRIDES = {
    # (company, product) -> end_year or None (for still active)
    ('SoFi', 'SoFi Crypto (Relaunched)'): None,  # relaunched 2025, still active
}

def find_disc_year(desc):
    if not desc or desc == 'nan':
        return None
    years = []
    for pat in _disc_patterns:
        for m in pat.finditer(desc):
            try:
                y = int(m.group(1))
                if 2000 <= y <= 2026:
                    years.append(y)
            except ValueError:
                pass
    return min(years) if years else None

df['end_year'] = None
for idx, r in df[df['entry_type'] == 'Product'].iterrows():
    key = (r['company_name'], r['product_name'])
    if key in DISC_YEAR_OVERRIDES:
        df.at[idx, 'end_year'] = DISC_YEAR_OVERRIDES[key]
        continue
    y = find_disc_year(str(r.get('product_history_and_description', '')))
    if y is not None:
        df.at[idx, 'end_year'] = y

n_discontinued = df['end_year'].notna().sum()
print(f"{n_discontinued} products flagged as discontinued (with end_year)")

# ── Build yearly active products panel ───────────────────────────────────────
# For each product that's kept in the universe (unflagged), create a row for
# each year it was active (start_year to end_year or current year).

import datetime
CURRENT_YEAR = datetime.datetime.now().year

# Filter to products that belong in the universe:
# - entry_type == 'Product'
# - not flagged as rebrand/feature/variant/excluded
# (We use the canonical name to avoid double-counting rebrand chains.)
universe = df[df['entry_type'] == 'Product'].copy()
for flag in ['is_rebrand', 'is_feature', 'is_variant', 'is_excluded', 'is_non_us']:
    universe = universe[~universe[flag].fillna(False)]

# Convert year_launched to int, skip products with no valid launch year
universe['year_launched_int'] = pd.to_numeric(universe['year_launched'], errors='coerce')
universe = universe[universe['year_launched_int'].notna()].copy()
universe['year_launched_int'] = universe['year_launched_int'].astype(int)

# Determine end year: use end_year if set, else current year
universe['end_year_int'] = universe['end_year'].apply(
    lambda x: int(x) if pd.notna(x) else CURRENT_YEAR)

# Cap weird start years (before 2000) at 2000 to keep panel manageable
# but keep them flagged as early-launch products for separate analysis
PANEL_START_YEAR = 2005

# Build panel: one row per (company, product, year)
panel_rows = []
for _, r in universe.iterrows():
    start = max(r['year_launched_int'], PANEL_START_YEAR)
    end = min(r['end_year_int'], CURRENT_YEAR)
    for y in range(start, end + 1):
        panel_rows.append({
            'year': y,
            'company_name': r['company_name'],
            'product_name': r['product_name'],
            'canonical_name': r['canonical_name'],
            'product_category': r['product_category'],
            'product_subcategory': r['product_subcategory'],
            'year_launched': r['year_launched_int'],
            'end_year': r['end_year_int'] if pd.notna(r['end_year']) else None,
        })

panel = pd.DataFrame(panel_rows)
print(f"\nBuilt panel: {len(panel):,} product-year observations "
      f"across {panel['year'].nunique()} years ({panel['year'].min()}-{panel['year'].max()})")
print(f"  Unique products: {universe.shape[0]}")
print(f"  Active products in {CURRENT_YEAR}: {(panel['year'] == CURRENT_YEAR).sum()}")
print(f"  Active products in 2020: {(panel['year'] == 2020).sum()}")
print(f"  Active products in 2015: {(panel['year'] == 2015).sum()}")
print(f"  Active products in 2010: {(panel['year'] == 2010).sum()}")

# Save panel
panel_file = os.path.join(DATA_CLEANED_DIR, 'fintech_product_panel.csv')
panel.to_csv(panel_file, index=False)
print(f"\nSaved panel to {panel_file}")

# ── Broad product category (4 groups) ───────────────────────────────────────
BROAD_CATEGORY = {
    # Savings (deposits + investments)
    'Banking (Consumer)': 'Savings',
    'Banking (Business)': 'Savings',
    'Investing': 'Savings',
    'Crypto/Digital Assets': 'Savings',
    'Treasury Management': 'Savings',
    'Alternative Investments': 'Savings',
    'Insurance & Benefits': 'Savings',
    # Borrowing
    'Lending (Consumer)': 'Borrowing',
    'Lending (Business)': 'Borrowing',
    'Credit Building': 'Borrowing',
    # Payments
    'Payments': 'Payments',
    'Money Transfer': 'Payments',
    'Bill Pay': 'Payments',
    'Payments API': 'Payments',
    'Point-of-Sale': 'Payments',
    # Credit Cards
    'Credit Cards': 'Credit Cards',
}
df['product_broad_category'] = df['product_subcategory'].map(BROAD_CATEGORY)

_bc_counts = df.loc[df['entry_type'] == 'Product', 'product_broad_category'].value_counts(dropna=False)
print(f"\nBroad product category distribution (products only):")
for cat, n in _bc_counts.items():
    label = cat if pd.notna(cat) else '(unmapped)'
    print(f"  {label:15s} {n:5d}")

# Save master merged CSV (with remapped subcategories, sorted by company name)
df.sort_values('company_name', key=lambda x: x.str.lower()).to_csv(
    os.path.join(DATA_CLEANED_DIR, 'fintech_timelines_master.csv'), index=False)
print(f"Saved {DATA_CLEANED_DIR}/fintech_timelines_master.csv")
