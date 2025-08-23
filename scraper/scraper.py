import sqlite3
import logging
import asyncio
import json
import csv
import os
# import yaml
from langdetect import detect
from deep_translator import GoogleTranslator
from playwright.async_api import async_playwright

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRAPING_TOPICS = {
    'database': {
        'search_query': 'database',
        'db_name': 'database/courses_database.db',
        'json_file': 'database/courses_database.json',
        'csv_file': 'database/courses_database.csv',
    },
    # 'comp_netw': {
    #     'search_query': 'computer%20networking',
    #     'db_name': 'comp_netw/courses_comp_netw.db',
    #     'json_file': 'comp_netw/courses_comp_netw.json',
    #     'csv_file': 'comp_netw/courses_comp_netw.csv',
    # },
    'artif_int': {
        'search_query': 'artificial%20intelligence',
        'db_name': 'aritf_int/courses_artif_int.db',
        'json_file': 'aritf_int/courses_artif_int.json',
        'csv_file': 'aritf_int/courses_artif_int.csv',
    },
    'big_data': {
        'search_query': 'big%20data',
        'db_name': 'big_data/courses_big_data.db',
        'json_file': 'big_data/courses_big_data.json',
        'csv_file': 'big_data/courses_big_data.csv',
    },
    'cloud_comp': {
        'search_query': 'cloud%20computing',
        'db_name': 'cloud_comp/courses_cloud_comp.db',
        'json_file': 'cloud_comp/courses_cloud_comp.json',
        'csv_file': 'cloud_comp/courses_cloud_comp.csv',
    },
    'blockchain': {
        'search_query': 'blockchain',
        'db_name': 'blockchain/courses_blockchain.db',
        'json_file': 'blockchain/courses_blockchain.json',
        'csv_file': 'blockchain/courses_blockchain.csv',
    },
    # 'netw_sec': {
    #     'search_query': 'network%20security',
    #     'db_name': 'netw_sec/courses_netw_sec.db',
    #     'json_file': 'netw_sec/courses_netw_sec.json',
    #     'csv_file': 'netw_sec/courses_netw_sec.csv',
    # },
}

class Database:
    """Database class to maintain a persistent connection and handle updates."""
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        """Create the courses table if it doesn't exist."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS courses (
                url TEXT PRIMARY KEY,
                original_title TEXT,
                translated_title TEXT,
                original_what_youll_learn TEXT,
                translated_what_youll_learn TEXT,
                original_skills_youll_gain TEXT,
                translated_skills_youll_gain TEXT,
                original_description_and_objectives TEXT,
                translated_description_and_objectives TEXT
            )
        ''')
        self.conn.commit()

    def insert_or_update_course(self, course_details):
        """Insert or update a course in the database."""
        self.cursor.execute(''' 
            INSERT INTO courses (url, original_title, translated_title, original_what_youll_learn, 
                translated_what_youll_learn, original_skills_youll_gain, translated_skills_youll_gain, 
                original_description_and_objectives, translated_description_and_objectives)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                original_title=excluded.original_title,
                translated_title=excluded.translated_title,
                original_what_youll_learn=excluded.original_what_youll_learn,
                translated_what_youll_learn=excluded.translated_what_youll_learn,
                original_skills_youll_gain=excluded.original_skills_youll_gain,
                translated_skills_youll_gain=excluded.translated_skills_youll_gain,
                original_description_and_objectives=excluded.original_description_and_objectives,
                translated_description_and_objectives=excluded.translated_description_and_objectives
        ''', (
            course_details["URL"], course_details["OriginalTitle"], course_details["TranslatedTitle"],
            ", ".join(course_details["OriginalWhatYoullLearn"]), ", ".join(course_details["TranslatedWhatYoullLearn"]),
            ", ".join(course_details["OriginalSkillsYoullGain"]), ", ".join(course_details["TranslatedSkillsYoullGain"]),
            course_details["OriginalDescriptionAndObjectives"], course_details["TranslatedDescriptionAndObjectives"]
        ))
        self.conn.commit()

    def remove_course(self, course_url):
        """Remove a course from the database."""
        self.cursor.execute('DELETE FROM courses WHERE url = ?', (course_url,))
        self.conn.commit()

    def get_all_courses(self):
        """Retrieve all courses from the database."""
        self.cursor.execute('SELECT * FROM courses')
        return self.cursor.fetchall()

    def close(self):
        """Close the database connection."""
        self.conn.close()

def translate_text(text):
    """Translate text into English."""
    try:
        detected_lang = detect(text)
        if detected_lang != 'en':
            return GoogleTranslator(source=detected_lang, target='en').translate(text)
        return text
    except Exception as e:
        logging.error(f"Error translating text: {e}")
        return text

async def update_json_and_csv(db, topic):
    """Update JSON and CSV files with the latest database state."""
    courses = db.get_all_courses()

    # Update JSON
    json_data = [
        {
            "url": course[0],
            "original_title": course[1],
            "translated_title": course[2],
            "original_what_youll_learn": course[3],
            "translated_what_youll_learn": course[4],
            "original_skills_youll_gain": course[5],
            "translated_skills_youll_gain": course[6],
            "original_description_and_objectives": course[7],
            "translated_description_and_objectives": course[8]
        }
        for course in courses
    ]
    with open(SCRAPING_TOPICS[topic]['json_file'], 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)

    # Update CSV
    with open(SCRAPING_TOPICS[topic]['csv_file'], 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "url", "original_title", "translated_title", "original_what_youll_learn",
            "translated_what_youll_learn", "original_skills_youll_gain",
            "translated_skills_youll_gain", "original_description_and_objectives",
            "translated_description_and_objectives"
        ])
        writer.writerows(courses)

async def scrape_course_details(course_url, browser, db, topic):
    """Scrape course details and update storage."""
    course_details = {
        "URL": course_url,
        "OriginalTitle": "",
        "TranslatedTitle": "",
        "OriginalWhatYoullLearn": [],
        "TranslatedWhatYoullLearn": [],
        "OriginalSkillsYoullGain": [],
        "TranslatedSkillsYoullGain": [],
        "OriginalDescriptionAndObjectives": "",
        "TranslatedDescriptionAndObjectives": ""
    }
    logging.info(f"Scraping course: {course_url}")
    try:
        page = await browser.new_page()
        await page.goto(course_url, timeout=60000)

        # Scrape course title
        try:
            title_locator = page.locator('h1[data-e2e="hero-title"]')
            if await title_locator.is_visible():
                original_title = await title_locator.text_content()
                course_details["OriginalTitle"] = original_title.strip()
                course_details["TranslatedTitle"] = translate_text(original_title.strip())
            else:
                logging.warning("Course title is missing.")
        except Exception as e:
            logging.warning(f"Error extracting course title: {e}")

        # Scrape "What You'll Learn"
        try:
            learn_locator = page.locator('div[data-track-component="what_you_will_learn_section"]')
            if await learn_locator.is_visible():
                items = await learn_locator.locator('li').all_text_contents()
                course_details["OriginalWhatYoullLearn"] = [item.strip() for item in items if item.strip()]
                course_details["TranslatedWhatYoullLearn"] = [
                    translate_text(item) for item in course_details["OriginalWhatYoullLearn"]
                ]
            else:
                logging.warning("What You'll Learn section is missing.")
        except Exception as e:
            logging.warning(f"Error extracting 'What You'll Learn': {e}")

        # Scrape "Skills You'll Gain"
        try:
            skills_locator = page.locator('a[data-click-key*="consumer_"][data-click-key*="_page.click.seo_skills_link_tag"]')
            if await skills_locator.count() > 0:
                items = await skills_locator.all_text_contents()
                course_details["OriginalSkillsYoullGain"] = [item.strip() for item in items if item.strip()]
                course_details["TranslatedSkillsYoullGain"] = [
                    translate_text(item) for item in course_details["OriginalSkillsYoullGain"]
                ]
            else:
                logging.warning("Skills You'll Gain section is missing.")
        except Exception as e:
            logging.warning(f"Error extracting 'Skills You'll Gain': {e}")

        # Scrape description and objectives
        try:
            # Fixed list of selectors
            selectors = [
                '#courses div div.content p',
                '#details div div.content p',
                '#modules div div.content p'
            ]

            desc_texts = []
            for selector in selectors:
                try:
                    # Locate and extract all <p> elements for the selector
                    desc_locator = page.locator(selector)
                    texts = await desc_locator.all_text_contents()
                    desc_texts.extend([text.strip() for text in texts if text.strip()])
                except Exception as inner_e:
                    logging.warning(f"Error with selector {selector}: {inner_e}")

            # Combine all extracted texts
            course_details["OriginalDescriptionAndObjectives"] = " ".join(desc_texts)

            if not course_details["OriginalDescriptionAndObjectives"]:
                logging.warning("Description and objectives section is missing.")
            else:
                course_details["TranslatedDescriptionAndObjectives"] = translate_text(
                    course_details["OriginalDescriptionAndObjectives"]
                )
        except Exception as e:
            logging.warning(f"Error extracting description: {e}")

        await page.close()

        # Insert or update course in database and storage
        db.insert_or_update_course(course_details)
        logging.info(f"Successfully updated course: {course_details['URL']}")
        await update_json_and_csv(db, topic)

    except Exception as e:
        logging.error(f"Error scraping course: {e}")

# Scrape all courses in a single page function
async def scrape_all_courses(base_url, db, topic):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        current_url = base_url
        num_pages = 3

        # Extract course links from the few first pages, if possible
        for i in range(1,num_pages+1):
            logging.info(f"Scraping page: {current_url}")
            await page.goto(current_url, timeout=60000)
            
            # Extract course links from the current page
            try:
                course_links = await page.locator('a[data-click-key="seo_entity_page.search.click.search_card"]').all()
                for course in course_links:
                    course_url = f"https://www.coursera.org{await course.get_attribute('href')}"
                    await scrape_course_details(course_url, browser, db, topic)
            except Exception as e:
                logging.warning(f"Error extracting course links: {e}")

            # Checking for next page and continuing scraping, until the final amount of targeted pages
            if i < num_pages:
                # Instead of using wait_for(), check if the element exists
                try:
                    # Modify the part where the program checks for the last page
                    next_button_disabled = await page.get_by_label('Next Page').is_disabled()

                    # Check if the element is visible (without a timeout)
                    if next_button_disabled:
                        logging.info("Target button element is disabled. Stopping scraping.")
                        break  # Stop scraping if the element is found
                    else:
                        logging.info("Target button element is not disabled, moving to next page.")
                except Exception as e:
                    logging.warning(f"Error checking for the target element: {e}")
                    logging.info("Stopping scraping.")
                    break
                
                # If the element is not found, increment the page number and continue scraping
                current_url = f"{current_url.split(f'page={i}')[0]}page={i+1}{current_url.split(f'page={i}')[1]}"
            
            await asyncio.sleep(2)  # Optional delay before scraping the next page

        await browser.close()

def display_existing_data(db):
    """Fetch and display all existing courses in the database."""
    courses = db.get_all_courses()
    if courses:
        print("\nPreexisting Data in the Database:")
        print("=" * 80)
        course_number = 1
        for course in courses:
            print(f"{course_number}. URL: {course[0]}")
            print(f"Original Title: {course[1]}")
            print(f"Translated Title: {course[2]}")
            print(f"Original What You'll Learn: {course[3]}")
            print(f"Translated What You'll Learn: {course[4]}")
            print(f"Original Skills You'll Gain: {course[5]}")
            print(f"Translated Skills You'll Gain: {course[6]}")
            print(f"Original Description and Objectives: {course[7]}")
            print(f"Translated Description and Objectives: {course[8]}")
            print("=" * 80)
            course_number += 1
    else:
        print("\nNo preexisting data found in the database.")

def main(config_path=str):
    """Main function."""
    # with open(config_path, 'r') as f:
    #     config = yaml.safe_load(f)
    
    for k in SCRAPING_TOPICS.keys():
        db = Database(SCRAPING_TOPICS[k]['db_name'])
        try:
            # Ensure JSON and CSV files exist
            if not os.path.exists(SCRAPING_TOPICS[k]['json_file']):
                with open(SCRAPING_TOPICS[k]['json_file'], 'w') as f:
                    json.dump([], f)
            if not os.path.exists(SCRAPING_TOPICS[k]['csv_file']):
                with open(SCRAPING_TOPICS[k]['csv_file'], 'w') as f:
                    pass

            # Display preexisting data
            display_existing_data(db)

            # Start scraping
            topic = SCRAPING_TOPICS[k]['search_query']
            base_url = f'https://www.coursera.org/courses?query={topic}&page=1&productDuration=1-3%20Months&productDuration=3-6%20Months'
            asyncio.run(scrape_all_courses(base_url, db, k))
        finally:
            db.close()

if __name__ == "__main__":
    # config_path = "scraper_config.yaml"
    # main(config_path)
    main()