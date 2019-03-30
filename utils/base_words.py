#-----#-----DEMOGRAPHIC WORDS-----#-----#
LEFT_WORDS = ["Democrat", "Democrats", "Democratic", "liberal", "liberals", 
"liberalism", "left wing", "Obama", "Biden", "Clinton", "Kaine", "Sanders", 
"Barack", "Hillary", "Bernie"]
RIGHT_WORDS = ["Republican", "Republicans", "conservative", "conservatives", 
"conservatism", "gop","right wing", "McCain", "Palin", "Ron Paul", "Mitt", "Romney", 
"Paul Ryan", "Gingrich", "Santorum", "Michelle Bachmann", "Donald", "Trump",
"Pence", "Kasich", "Cruz", "Rubio", "Carson","Bush", "Fiorina"]


MALE_WORDS = ["man", "men", "male", "boy"]
FEMALE_WORDS = ["woman", "women", "female", "girl"]


WHITE_WORDS = ["white"]
WHITE_WORDS_EXCLUDE=["house"]
MINORITY_WORDS = ["black","mexican","asian","muslim","african","latino","latina"]



#-----#-----BANNED WORDS-----#-----#
NEWS =["advertis_continu", "box_invalid", "read_main", "email_address", "servic_thank", "sign_receiv", "special_offer", "robot_click", "contribut_writer",
"subscrib_error","tri_later","newslett_subscrib","new_york", 'pleas_robot', 'click_box', 'invalid_email', 'address_pleas', 'select_newslett', 'subscrib_sign', 
'receiv_occasion', 'updat_special', 'offer_new', 'york_time', 'product_servic', 'thank_subscrib', 'error_occur', 'pleas_tri','pleas_tri_later', 'later_view', 'new_york', 'time_newslett',"today_newslett",
"front_page", "advertis", "new","york","pleas","advertis", "continu", "box","invalid", "read","main", "servic","thank", "sign","receiv", "special","offer", "robot","click", "contribut","writer",
"subscrib","error","tri","later","newslett","subscrib","new","york", 'pleas','robot', 'click','box', 'invalid', 'address','pleas', 'select','newslett', 'subscrib','sign', 
'receiv','occasion', 'updat','special', 'offer','new', 'york','time', 'product','servic', 'thank','subscrib', 'error','occur', 'pleas','tri', 'later','view', 'new','york', 'time','newslett',"today','newslett",
"front","page","unit_state","tim","pleas_verifi","inbox_monday_friday","invalid_email_address","inbox_monday",
"deliv_inbox_monday","stori", "commentari", "columnist", "editori", "nyt", "cnn", "pdf","txt","newslett_enter","valid_email","newslett_signup","address_field"
"signup_center","pleas_click","visit_newslett","first_newslett","soon_sorri","regist_newslett","error_process","newslett_receiv","begin_receiv","pleas_enter","polit_digest",
"digest_newslett"]

NAMES=["rodham","georg","ryan","obama","clinton","michel","john","marco","cantor","ann","adelson",
		"ted","christi","mitch","mcconnel","lindsey","graham","jindal","rand","paul","cameron", "eric",
		"governor","cheney","cole","susan","collin","senat","richard","mourdock","perri","charl",
		"tom","todd","bobbi","chri","christi","eric","bachmann","romney","todd","akin","gore","snyder","jeb",
		"diann","feinstein","kerri","brennan","warren","elizabeth","lyndon","pelosi",
		"tea","palin","sarah","bush","kushner","flynn","ivanka","stephen","manafort","conway","strategist",
		"jeff","kellyann","conway","steve","bannon","bob","mike","koch","pompeo","sean","spicer","friedman",
		"corker","boehner","devin","jeff","sessions","rex","tillerson","trump","chuck","schumer","nanci","tim",
		"adam","kennedi","wasserman","warner","nelson","christoph","loretta","lynch","harri","benjamin"]

TITLES=["secretari","governor","chairman","gov","vice","prime_minist","prime","speaker","presid","congresswoman"
		"administr","foundat"]

OTHER=["say"]

BANNED = NEWS+NAMES+TITLES


MONTH_NAMES = ["january","february","march","april","may","june","july","august","september","october","november","december"]
