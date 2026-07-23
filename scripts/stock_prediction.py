from __future__ import annotations

from datetime import date, timedelta
from math import ceil, isfinite
from typing import Any


def _display_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _as_non_negative_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name} must be a finite number")
    if number < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return number


def _validate_sales(values: Any) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError("sales_last_7_days must be a non-empty list")

    sales: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"sales_last_7_days[{index}] must be a number")
        number = float(value)
        if not isfinite(number):
            raise ValueError(f"sales_last_7_days[{index}] must be a finite number")
        if number < 0:
            raise ValueError(f"sales_last_7_days[{index}] must be >= 0")
        sales.append(number)
    return sales


def invalid_stock_prediction(message: str) -> dict[str, Any]:
    return {
        "success": False,
        "message": f"Invalid input: {message}",
    }


def predict_stock_depletion(payload: dict[str, Any], today: date | None = None) -> dict[str, Any]:
    try:
        if not isinstance(payload, dict):
            raise ValueError("body must be a JSON object")

        product_id = str(payload.get("product_id") or "").strip()
        if not product_id:
            raise ValueError("product_id is required")

        product_title = str(payload.get("product_title") or "").strip()
        current_stock = _as_non_negative_number(payload.get("current_stock"), "current_stock")
        lead_time_days = _as_non_negative_number(payload.get("lead_time_days", 0), "lead_time_days")
        safety_stock = _as_non_negative_number(payload.get("safety_stock", 0), "safety_stock")
        sales = _validate_sales(payload.get("sales_last_7_days"))

        average_daily_sales = round(sum(sales) / len(sales), 2)
        if average_daily_sales == 0:
            return {
                "success": True,
                "product_id": product_id,
                "product_title": product_title,
                "current_stock": _display_number(current_stock),
                "average_daily_sales": 0,
                "days_until_stockout": None,
                "estimated_stockout_date": None,
                "risk_level": "low",
                "reorder_now": False,
                "recommended_reorder_quantity": 0,
                "message": "Aucune vente récente détectée. Le risque d’épuisement est faible.",
            }

        days_until_stockout = round(current_stock / average_daily_sales, 2)
        today_value = today or date.today()
        stockout_date = today_value + timedelta(days=days_until_stockout)
        stockout_day_delta = (stockout_date - today_value).days

        if current_stock <= safety_stock or stockout_day_delta <= lead_time_days:
            risk_level = "high"
            message = "Stock critique. Réapprovisionnement immédiat recommandé."
        elif stockout_day_delta <= lead_time_days + 3:
            risk_level = "medium"
            message = "Stock moyen. Réapprovisionnement conseillé."
        else:
            risk_level = "low"
            message = "Stock suffisant. Aucun réapprovisionnement immédiat requis."

        recommended_reorder_quantity = ceil((average_daily_sales * 14) + safety_stock - current_stock)

        return {
            "success": True,
            "product_id": product_id,
            "product_title": product_title,
            "current_stock": _display_number(current_stock),
            "average_daily_sales": average_daily_sales,
            "days_until_stockout": days_until_stockout,
            "estimated_stockout_date": stockout_date.isoformat(),
            "risk_level": risk_level,
            "reorder_now": risk_level in {"high", "medium"},
            "recommended_reorder_quantity": max(recommended_reorder_quantity, 0),
            "message": message,
        }
    except ValueError as exc:
        return invalid_stock_prediction(str(exc))
    except Exception as exc:
        return {
            "success": False,
            "message": f"Invalid input: unexpected stock prediction error: {exc}",
        }
