import pathlib
import sys

import pandas as pd
import requests
from msal import ConfidentialClientApplication


def get_sso_auth_token():
    client_id = "f5419d3f-b70b-4f55-9113-181965725ff3"
    authority = (
        "https://login.microsoftonline.com/0753c1a4-2be6-4a86-8763-32ae847e1186"
    )
    client_secret = "xkL8Q~T2yorsApXXcIzkSH5j-RCgok6h45KFBcH7"
    scopes = ["api://d166b705-f663-4e1d-8ad6-457075f0f6fa/.default"]

    app = ConfidentialClientApplication(
        client_id=client_id,
        authority=authority,
        client_credential=client_secret,
    )

    result = app.acquire_token_for_client(scopes=scopes)

    try:
        if result:
            return result["access_token"]
        else:
            raise Exception("Auth-Token: Not able to fetch access token.")
    except Exception as e:
        print("Auth-Token: Unable to acquire auth token.", e)


def _base_dir() -> pathlib.Path:
    """
    Resolve a base directory that works both in development and in a
    frozen (PyInstaller) build.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # Running from a PyInstaller bundle
        return pathlib.Path(sys._MEIPASS)  # type: ignore[attr-defined]

    # Editable / normal install: project root = three levels up from this file
    return pathlib.Path(__file__).resolve().parent


def _resolve_options_codes_path() -> pathlib.Path:
    """
    Try to locate Options_codes.txt relative to the application:
    - Prefer {base_dir}/assets/Options_codes.txt
    - Fallback to {base_dir}/Options_codes.txt
    """
    base = _base_dir()

    assets_path = base / "assets" / "Options_codes.txt"
    if assets_path.exists():
        return assets_path

    root_path = base / "Options_codes.txt"
    return root_path


def get_instruments_data_from_api():
    file_path = _resolve_options_codes_path()

    available_product_ids = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) != 2:
                continue

            product_id = parts[1].strip()
            if product_id.isdigit():
                available_product_ids.append(product_id)

    # remove duplicates (safe)
    available_product_ids = list(dict.fromkeys(available_product_ids))

    print(len(available_product_ids))
    print(
        "[Instruments-Data-Feed]: All app registered product_ids are fetched "
        "successfully."
    )

    instruments_data = []
    chunk_size = 20

    for i in range(0, len(available_product_ids), chunk_size):
        chunk = available_product_ids[i : i + chunk_size]
        product_id_query = ", ".join(chunk)

        query_body = f"""
            query {{
                instrument(hg_product_id: "{product_id_query}") {{
                    combo_type
                    currency
                    currency_code
                    denom_main_fraction
                    denom_sub_fraction
                    display_factor
                    exchange_code
                    exchange_id
                    expiry_date
                    hg_expiry_date
                    hg_instrument_id
                    hg_product_id
                    implied_market_depth
                    instrument_alias
                    instrument_name
                    is_active
                    is_expired
                    is_inter_product
                    is_tas
                    last_trade_date
                    legs
                    market_depth
                    market_segment_id
                    market_subgroup
                    market_subgroup_code
                    matching_algorithm
                    maturity_date
                    max_trade_vol
                    mic_code
                    min_lot_size
                    min_trade_vol
                    option_code
                    option_scheme
                    point_value
                    price_display_decimals
                    price_display_format
                    price_display_type
                    product_code
                    product_family_code
                    product_family_id
                    product_name
                    qty_of_measure
                    quant_code
                    ric_code
                    round_lot_qty
                    security_id
                    security_type
                    security_type_code
                    series
                    series_term
                    source
                    source_platform
                    start_date
                    strike_price
                    sub_exchange_code
                    supports_implieds
                    tick_denominator
                    tick_numerator
                    tick_rule
                    tick_size
                    tick_value
                    tt_instrument_id
                    underlying_hg_instrument_id
                    underlying_tas_id
                    unit_of_measure
                    update_date
                    version_id
                    working_days_to_expire
                }}
            }}

        """

        try:
            auth_token = get_sso_auth_token()
            response = requests.post(
                "https://ng-api.prod-live.hertshtengroup.com/pds/graphql/",
                json={"query": query_body},
                verify=False,
                headers={"Authorization": f"Bearer {auth_token}"},
            )

            response.raise_for_status()
            response_data = response.json()

            if "data" not in response_data or "instrument" not in response_data["data"]:
                raise ValueError(
                    "Unexpected response format: 'instrument' key missing"
                )

            instruments_data.extend(response_data["data"]["instrument"])

        except Exception as e:
            print(
                "[Instruments-Data-Feed]: Error while fetching instruments data:",
                e,
            )

    instruments_df = pd.DataFrame(instruments_data)

    if instruments_df.empty:
        raise ValueError("No data returned from API")

    # Return in-memory DataFrame without writing to disk to keep things fast.
    return instruments_df


__all__ = ["get_instruments_data_from_api"]

