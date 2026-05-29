import time
import pandas as pd
import requests

# --- הגדרות ראשוניות ---
API_KEY = "YOUR_API_KEY_HERE"  # השתל את מפתח ה-API שלך כאן
SEARCH_QUERY = "שווארמה בישראל"  # השאילתה שלך
OUTPUT_FILE = "shawarma_places.csv"  # שם קובץ הפלט שיקלט


def fetch_all_places(query, api_key):
    url = "https://places.googleapis.com/v1/places:searchText"

    # הגדרת הכותרות (Headers). שים לב ל-FieldMask שמגדיר אילו שדות יחזרו (וחוסך כסף)
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.rating,places.userRatingCount,nextPageToken",
    }

    # גוף הבקשה הראשונית
    body = {"textQuery": query, "languageCode": "he"}

    all_places = []
    page_count = 1

    print(f"🔎 מתחיל לחפש: '{query}'...")

    while True:
        print(f"⏳ מושך נתונים מעמוד {page_count}...")
        response = requests.post(url, headers=headers, json=body)

        # בדיקת שגיאות בקשה
        if response.status_code != 200:
            print(f"❌ שגיאה מהשרת ({response.status_code}): {response.text}")
            break

        data = response.json()
        places_in_page = data.get("places", [])

        # לקיחת הנתונים הרלוונטיים מכל מקום שנמצא
        for place in places_in_page:
            place_info = {
                "שם העסק": place.get("displayName", {}).get("text", "ללא שם"),
                "כתובת": place.get("formattedAddress", "ללא כתובת"),
                "קו רוחב (Latitude)": place.get("location", {}).get("latitude"),
                "קו אורך (Longitude)": place.get("location", {}).get(
                    "longitude"
                ),
                "דירוג": place.get("rating", "אין"),
                "מספר מדרגים": place.get("userRatingCount", 0),
            }
            all_places.append(place_info)

        print(f"✅ נמצאו {len(places_in_page)} שווארמיות בעמוד זה.")

        # בדיקה האם יש עמוד תוצאות נוסף
        next_page_token = data.get("nextPageToken")
        if next_page_token:
            # מעדכנים את גוף הבקשה לפעם הבאה עם הטוקן שקיבלנו
            body["pageToken"] = next_page_token
            page_count += 1
            # גוגל ממליצה להמתין שנייה קלה בין בקשות דפדוף כדי שהטוקן יהיה אקטיבי בשרת שלהם
            time.sleep(1.5)
        else:
            # אין יותר עמודים, יוצאים מהלולאה
            print("🏁 אין עמודים נוספים. החיפוש הושלם!")
            break

    return all_places


# הרצת הפונקציה
if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY_HERE":
        print(
            "❌ עצור! אתה חייב להחליף את 'YOUR_API_KEY_HERE' במפתח ה-API האמיתי שלך מגוגל קלאוד."
        )
    else:
        results = fetch_all_places(SEARCH_QUERY, API_KEY)

        if results:
            # הפיכת רשימת המקומות לטבלה ושמירה כקובץ Excel/CSV
            df = pd.DataFrame(results)
            # encoding='utf-8-sig' מבטיח שהעברית לא תהפוך לג'יבריש באקסל
            df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
            print(f"\n🎉 הצלחה! נשמרו {len(results)} שווארמיות בקובץ: {OUTPUT_FILE}")
        else:
            print("\n לא נמצאו תוצאות לשמירה.")