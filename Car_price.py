import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and preprocessing files
try:
    model = joblib.load("car_price_model.joblib")
    scaler = joblib.load("scaler.joblib")
    encoder = joblib.load("encoded_columns.joblib")
    model_features = joblib.load("model_features.joblib")
except FileNotFoundError:
    st.error("Error: Model or preprocessing files not found. Please ensure all files are in the correct directory.")
    st.stop()

# Define hardcoded lists for categorical inputs
brands = [
    "BMW", "Ford", "Mercedes-Benz", "Chevrolet", "Toyota", "Audi", "Lexus", "Jeep", "Porsche", "Land", "Nissan",
    "Cadillac", "RAM", "GMC", "Dodge", "Kia", "Hyundai", "Subaru", "Acura", "Honda", "INFINITI", "Mazda", "Volkswagen",
    "Lincoln", "Jaguar", "Volvo", "MINI", "Buick", "Maserati", "Chrysler", "Mitsubishi", "Genesis", "Alfa", "Hummer",
    "Pontiac", "Bentley", "Scion", "Saturn", "Aston", "FIAT", "Lotus", "Mercury", "Rolls-Royce", "Saab", "Plymouth", "smart",
    "Maybach", "Suzuki"
]
models = [
    "Others", "F-TYPE R", "F-150 XLT", "Q5 2.0T Premium Plus", "AMG C 43 Base 4MATIC", "Z4 sDrive28i", "Sprinter 2500",
    "H2 Base", "911 Carrera S", "CLA-Class CLA 250", "Mustang GT Premium", "IS 350 Base", "M4 Base", "Cayenne Base",
    "GLS 450 Base 4MATIC", "M6 Base", "LC 500 Base", "A5 2.0T Premium", "GT-R Premium", "X7 xDrive40i", "Transit Connect XLT",
    "LX 570 Three-Row", "Cayman Base", "M5 Base", "4Runner SR5", "2500 Laramie", "Expedition Platinum", "Expedition Limited",
    "Highlander XLE", "Wrangler Unlimited Sahara", "Sierra 1500 SLT", "Transit-350 XLT", "XTS Luxury", "Land Cruiser Base",
    "1500 Laramie", "Corvette Base", "GX 470 Base", "Explorer XLT", "Telluride SX", "Q5 S line Premium Plus", "GX 460 Base",
    "1500 Big Horn", "X6 xDrive40i", "Durango GT", "Yukon XL SLT", "Ghibli S Q4", "4Runner Limited", "Corvette Z06",
    "M3 Base", "IS 250 Base", "ES 350 Base", "Excursion Limited", "SL-Class SL500 Roadster", "SL-Class SL 550", "Fusion SE",
    "QX80 Base", "Rover Range Rover Sport HSE", "Continental GT Base", "Escalade ESV Platinum", "Gladiator Overland",
    "Grand Cherokee L Limited", "X6 M Base", "Navigator Reserve", "Camaro 1LT", "Macan S", "Sequoia SR5",
    "GLE 350 Base 4MATIC", "Tahoe LTZ", "Tahoe LT", "Suburban LT", "GLC 300 GLC 300", "Tundra SR5", "Wrangler Unlimited Sport",
    "Countryman Cooper S ALL4", "RX 450h Base", "328 i xDrive", "740 i", "Grand Cherokee Limited", "Yukon Denali", "435 i",
    "E-Class E 350 4MATIC", "Camaro 1SS", "650 i", "Mustang EcoBoost Premium", "AMG G 63 Base", "Challenger R/T",
    "Sequoia Limited", "Wrangler Sahara", "A3 2.0T Premium", "A4 2.0T Premium", "Cayenne S", "F-150 Platinum",
    "E-Class E 350", "535 i xDrive", "F-150 Lariat", "F-250 Lariat", "S-Class S 63 AMG", "S-Class S 560 4MATIC",
    "Rover Range Rover Sport Supercharged", "Armada Platinum", "RX 350 Base", "Metris Base", "Transit-350 Base", "911 Carrera",
    "X5 xDrive35i", "Yukon SLT", "335 i", "Golf GTI 2.0T SE 4-Door", "Suburban 1500 LT", "Challenger SRT Hellcat",
    "FJ Cruiser Base", "Focus SE", "GX 460 Premium", "Cooper S Base", "Solstice GXP", "750 i", "G-Class G 550 4MATIC",
    "Wrangler Sport", "X5 xDrive50i", "Macan Base", "Suburban Premier", "F-150 Raptor", "S4 3.0T Premium Plus",
    "X5 M Base", "X5 xDrive40i", "H3 Base", "Corvette Stingray w/1LT", "Corvette Stingray w/2LT", "Yukon XL Denali",
    "C-Class C 300 4MATIC", "1500 Rebel", "X3 xDrive30i", "Cooper Base", "Sprinter 2500 Standard Roof", "330 i xDrive",
    "328 i", "F-150 XL", "911 Carrera 4S", "Corvette Stingray w/3LT", "Civic EX", "LS 460 Base", "Wrangler Unlimited Rubicon"
]
fuel_types = ['Gasoline', 'Hybrid', 'Others', 'Diesel']
transmissions = ['automatic', 'dual clutch', 'manual', 'cvt', 'other']
ext_colors = [
    "–", "Agate Black Metallic", "Alfa White", "Alpine White", "Anodized Blue Metallic", "Antimatter Blue Metallic",
    "Apex Blue", "Arctic Gray Metallic", "Arctic White", "Atomic Silver", "Aurora Black", "Baltic Gray", "Barcelona Red",
    "Beige", "Billet Clearcoat Metallic", "Billet Silver Metallic Clearcoat", "Black", "Black Cherry", "Black Clearcoat",
    "Black Forest Green", "Black Noir Pearl", "Black Obsidian", "Black Raven", "Black Sapphire Metallic", "Blue",
    "Blue Reflex Mica", "Blueprint", "Bright White Clearcoat", "Bronze Dune Metallic", "Brown", "Burnished Bronze Metallic",
    "Cajun Red Tintcoat", "Carbon Black Metallic", "Carbonized Gray Metallic", "Carrara White Metallic", "Caspian Blue",
    "Caviar", "Chalk", "Chronos Gray", "Chronos Gray Metallic", "Cirrus Silver Metallic", "Cobra Beige Metallic",
    "Crimson Red Tintcoat", "Crystal Black", "Crystal Black Pearl", "Crystal Black Silica", "Crystal White Pearl",
    "Dark Ash Metallic", "Dark Graphite Metallic", "Dark Matter Metallic", "Dark Moon Blue Metallic", "Dark Moss",
    "Daytona Gray", "Daytona Gray Pearl Effect", "Dazzling White", "DB Black Clearcoat", "Deep Black Pearl Effect",
    "Deep Black Pearl Effect / Black Roof", "Delmonico Red Pearlcoat", "designo Diamond White", "designo Diamond White Bright",
    "Diamond Black", "Donington Grey Metallic", "Ebony Black", "Ebony Twilight Metallic", "Eiger Grey", "Eiger Grey Metallic",
    "Electric Blue Metallic", "Emin White", "Eminent White Pearl", "Ember Pearlcoat", "Emerald Green Metallic",
    "Firecracker Red Clearcoat", "Firenze Red", "Firenze Red Metallic", "Flame Red Clearcoat", "Florett Silver",
    "Frozen White", "Fuji White", "Garnet Red Metallic", "Gecko Pearlcoat", "Glacial White Pearl", "Glacier Blue Metallic",
    "Glacier Silver Metallic", "Glacier White", "Glacier White Metallic", "Go Mango!", "Gold", "Granite Crystal Clearcoat Metallic",
    "Granite Crystal Metallic Clearcoat", "Graphite Grey", "Graphite Grey Metallic", "Gray", "Green", "Gun Metallic",
    "Hampton Gray", "Hellayella Clearcoat", "Horizon Blue", "Hydro Blue Pearlcoat", "Hyper Red", "Ibis White",
    "Ice Silver Metallic", "Imperial Blue Metallic", "Infrared Tintcoat", "Ingot Silver Metallic", "Iridescent Pearl Tricoat",
    "Iridium Metallic", "Ironman Silver", "Isle of Man Green Metallic", "Jet Black Mica", "Jungle Green", "Jupiter Red",
    "Kinetic Blue", "Liquid Platinum", "Lunar Blue Metallic", "Lunar Rock", "Lunar Silver Metallic", "Lunare White Metallic",
    "Machine Gray Metallic", "Magnetic Black", "Magnetic Gray Clearcoat", "Magnetic Gray Metallic", "Magnetic Metallic",
    "Magnetite Gray Metallic", "Majestic Black Pearl", "Majestic Plum Metallic", "Mango Tango Pearlcoat",
    "Manhattan Noir Metallic", "Maroon", "Matador Red Metallic", "Matador Red Mica", "Maximum Steel Metallic", "Metallic",
    "Midnight Black", "Midnight Black Metallic", "Midnight Blue Metallic", "Mineral White", "Mosaic Black Metallic",
    "Mountain Air Metallic", "Mythos Black", "Mythos Black Metallic", "Nautical Blue Pearl", "Navarra Blue", "Navarre Blue",
    "Nebula Gray Pearl", "Nightfall Gray Metallic", "Northsky Blue Metallic", "Obsidian", "Obsidian Black Metallic",
    "Octane Red Pearlcoat", "Onyx Black", "Orca Black Metallic", "Orange", "Oryx White Prl", "Oxford White", "Pacific Blue",
    "Pacific Blue Metallic", "Passion Red", "Pearl White", "Phantom Black", "Phantom Black Pearl Effect / Black Roof",
    "Phytonic Blue Metallic", "Pink", "Platinum Gray Metallic", "Platinum Quartz Metallic", "Platinum White Pearl",
    "Polymetal Gray Metallic", "Portofino Blue Metallic", "Portofino Gray", "Pristine White", "Pure White", "Quartz Blue Pearl",
    "Quartzite Grey Metallic", "Quicksilver Metallic", "Radiant Red Metallic II", "Red", "Red Obsession", "Red Quartz Tintcoat",
    "Redline Red", "Reflex Silver", "Remington Red Metallic", "Rich Garnet Metallic", "Ruby Flare Pearl",
    "Ruby Red Metallic Tinted Clearcoat", "Sangria Red", "Santorin Black", "Santorini Black Metallic", "Satin Steel Metallic",
    "Scarlet Ember", "Selenite Gray Metallic", "Selenite Grey Metallic", "Shadow Black", "Shadow Gray Metallic",
    "Shimmering Silver", "Silky Silver", "Silver", "Silver Flare Metallic", "Silver Ice Metallic", "Silver Mist",
    "Silver Radiance", "Silver Zynith", "Siren Red Tintcoat", "Snow White Pearl", "Snowflake White Pearl",
    "Snowflake White Pearl Metallic", "Sonic Silver Metallic", "Soul Red Crystal Metallic", "Sparkling Silver",
    "Stellar Black Metallic", "Sting Gray Clearcoat", "Stone Gray Metallic", "Stormy Sea", "Summit White",
    "Sunset Drift Chromaflair", "Super Black", "Super White", "Tango Red Metallic", "Titanium Silver", "Tungsten Metallic",
    "Twilight Black", "Twilight Blue Metallic", "Typhoon Gray", "Ultra Black", "Ultra White", "Velvet Red Pearlcoat", "Verde",
    "Vik Black", "Volcanic Orange", "Volcano Grey Metallic", "Vulcano Black Metallic", "White", "White Diamond Tri-Coat",
    "White Frost Tri-Coat", "White Knuckle Clearcoat", "White Platinum Tri-Coat Metallic", "Wind Chill Pearl", "Wolf Gray",
    "Yellow", "Yulong", "Yulong White"
]
int_colors = [
    "–", "Adrenaline Red", "Almond Beige", "Amber", "Aragon Brown", "Ash", "Beige", "Black", "Black / Graphite",
    "Black / Gray", "Black / Saddle", "Black Onyx", "Black w/Red Stitching", "Black/Gun Metal", "Black/Graphite", "Blue",
    "Boulder", "Brown", "Camel Leather", "Canberra Beige", "Canberra Beige/Black", "Carbon Black", "Ceramic", "Charcoal",
    "Charcoal Black", "Chateau", "Chestnut", "Classic Red", "Cloud", "Dark Auburn", "Dark Chestnut", "Dark Galvanized",
    "Dark Gray", "Deep Chestnut", "Deep Cypress", "Diesel Gray / Black", "Ebony", "Ebony Black", "Ebony / Ebony Accents",
    "Ebony/Light Oyster Stitch", "Espresso", "Gideon", "Global Black", "Gold", "Graphite", "Graphite w/Gun Metal", "Gray",
    "Graystone", "Green", "Ice", "Jet Black", "Kyalami Orange", "Light Gray", "Light Platinum / Jet Black", "Light Slate",
    "Light Titanium", "Macchiato", "Macchiato Beige/Black", "Macchiato/Magmagrey", "Magma Red", "Medium Ash Gray",
    "Medium Dark Slate", "Medium Earth Gray", "Medium Light Camel", "Medium Pewter", "Medium Stone", "Mistral Gray / Raven",
    "Mocha", "Mountain Brown", "Navy Pier", "Nero", "Nougat Brown", "Orange", "Orchid", "Oyster W/Contrast", "Oyster/Black",
    "Parchment", "Pearl Beige", "Pimento Red w/Ebony", "Platinum", "Red", "Red / Black", "Red/Black", "Rhapsody Blue",
    "Rioja Red", "Roast", "Rock Gray", "Saddle Brown", "Saiga Beige", "Sand Beige", "Sandstone", "Sarder Brown", "Satin Black",
    "Shale", "Shara Beige", "Silk Beige/Black", "Silk Beige/Espresso Brown", "Silver", "Slate", "Sport", "Tan", "Tan/Ebony",
    "Tan/Ebony/Ebony", "Tension", "Titan Black", "Titan Black / Quarzit", "Tupelo", "Very Light Cashmere", "Walnut",
    "Whisper Beige", "White", "White / Brown", "Yellow"
]
accident_status = ['None', 'Yes']

# Conversion rate: 1 USD = 280 PKR
USD_TO_PKR = 280

# Streamlit app layout
st.title("🚗 Car Price Prediction App")
st.write("Fill in the details below to predict your car's price in PKR.")

# Form layout
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", brands)
        model_ = st.selectbox("Model", models)
        fuel_type = st.selectbox("Fuel Type", fuel_types)
        transmission = st.selectbox("Transmission", transmissions)

    with col2:
        ext_col = st.selectbox("Exterior Color", ext_colors)
        int_col = st.selectbox("Interior Color", int_colors)
        accident = st.selectbox("Any Accidents?", accident_status)

    model_year = st.number_input("Model Year", min_value=1990, max_value=2025, value=2020)
    milage = st.number_input("Mileage (Miles)", min_value=0, max_value=500000, value=50000)
    engine_hp = st.number_input("Horsepower (HP)", min_value=30, max_value=1500, value=200)
    engine_liters = st.number_input("Engine Size (Liters)", min_value=0.5, max_value=10.0, step=0.1, value=2.0)

    submit = st.form_submit_button("Predict Price")

# Prediction logic
if submit:
    # Create input dictionary with correct column names
    input_dict = {
        "brand": brand,
        "model": model_,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "ext_col": ext_col,
        "int_col": int_col,
        "accident": accident,
        "model_year": model_year,
        "milage": milage,
        "engine_hp": engine_hp,
        "engine_liters": engine_liters
    }

    # Convert input to DataFrame
    df_input = pd.DataFrame([input_dict])

    try:
        # Transform input using the saved preprocessor and model
        transformed_input = model.named_steps['preprocessor'].transform(df_input)
        
        # Ensure the transformed input matches the model's expected features
        df_encoded = pd.DataFrame(
            transformed_input,
            columns=model_features
        )

        # Predict price in USD
        prediction_usd = model.named_steps['model'].predict(df_encoded)[0]
        
        # Convert to PKR
        prediction_pkr = prediction_usd * USD_TO_PKR
        prediction_pkr = np.round(prediction_pkr, 0)

        st.success(f"💰 Estimated Car Price: PKR {prediction_pkr:,.0f}")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")