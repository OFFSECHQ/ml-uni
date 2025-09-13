Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Some Message",
        autoProcessQueue: false
    });
    
    dz.on("addedfile", function() {
        if (dz.files[1]!=null) {
            dz.removeFile(dz.files[0]);        
        }
    });

    dz.on("complete", function (file) {
        let imageData = file.dataURL;
        
        // Show loading state
        $("#submitBtn").addClass("btn-loading").prop("disabled", true);
        $("#submitBtn .btn-text").hide();
        $("#submitBtn .btn-loading-text").show();
        $("#error").hide();
        $("#resultHolder .empty-state").hide();
        $("#divClassTable").hide();
        
        var url = "http://127.0.0.1:5000/classify_image";

        $.post(url, {
            image_data: file.dataURL
        },function(data, status) {
            /* 
            Below is a sample response if you have two faces in an image lets say virat and roger together.
            Most of the time if there is one person in the image you will get only one element in below array
            data = [
                {
                    class: "viral_kohli",
                    class_probability: [1.05, 12.67, 22.00, 4.5, 91.56],
                    class_dictionary: {
                        lionel_messi: 0,
                        maria_sharapova: 1,
                        roger_federer: 2,
                        serena_williams: 3,
                        virat_kohli: 4
                    }
                },
                {
                    class: "roder_federer",
                    class_probability: [7.02, 23.7, 52.00, 6.1, 1.62],
                    class_dictionary: {
                        lionel_messi: 0,
                        maria_sharapova: 1,
                        roger_federer: 2,
                        serena_williams: 3,
                        virat_kohli: 4
                    }
                }
            ]
            */
            console.log(data);
            
            // Remove loading state
            $("#submitBtn").removeClass("btn-loading").prop("disabled", false);
            $("#submitBtn .btn-text").show();
            $("#submitBtn .btn-loading-text").hide();
            
            if (!data || data.length==0) {
                $("#resultHolder .empty-state").show();
                $("#divClassTable").hide();                
                $("#error").show();
                return;
            }
            let players = ["christiano_ronaldo", "gukesh", "neeraj_chopra", "pv_sindhu", "shubhman_gill"];
            
            let match = null;
            let bestScore = -1;
            for (let i=0;i<data.length;++i) {
                let maxScoreForThisClass = Math.max(...data[i].class_probability);
                if(maxScoreForThisClass>bestScore) {
                    match = data[i];
                    bestScore = maxScoreForThisClass;
                }
            }
            if (match) {
                $("#error").hide();
                $("#resultHolder .empty-state").hide();
                $("#resultHolder").show();
                $("#divClassTable").show();

                let playerClass = match.class;
                let playerName = "";
                let imageSrc = "";

                if (playerClass === 'chritiano_ronaldo') {
                    playerName = 'Christiano Ronaldo';
                    imageSrc = './IMAGES/ronaldo.jpg';
                } else if (playerClass === 'gukesh') {
                    playerName = 'Gukesh Dommaraju';
                    imageSrc = './IMAGES/gukesh.jpg';
                } else if (playerClass === 'neeraj_chopra') {
                    playerName = 'Neeraj Chopra';
                    imageSrc = './IMAGES/neeraj.jpg';
                } else if (playerClass === 'pv_sindhu') {
                    playerName = 'PV Sindhu';
                    imageSrc = './IMAGES/pv.jpg';
                } else if (playerClass === 'shubhman_gill') {
                    playerName = 'Shubman Gill';
                    imageSrc = './IMAGES/gill.jpg';
                }

                $("#resultHolder").html(`
                    <div class="card-body">
                        <img src="${imageSrc}" alt="Player Image" class="img-fluid rounded mb-3">
                        <h5 class="card-title">${playerName}</h5>
                    </div>
                `);
                
                let classDictionary = match.class_dictionary;
                for(let personName in classDictionary) {
                    let index = classDictionary[personName];
                    let proabilityScore = match.class_probability[index];
                    let elementName = "#score_" + personName;
                    $(elementName).html(proabilityScore.toFixed(2) + "%");
                }
            }
            // dz.removeFile(file);            
        }).fail(function() {
            // Remove loading state on error
            $("#submitBtn").removeClass("btn-loading").prop("disabled", false);
            $("#submitBtn .btn-text").show();
            $("#submitBtn .btn-loading-text").hide();
            $("#resultHolder .empty-state").show();
            $("#divClassTable").hide();
            $("#error").show();
        });
    });

    $("#submitBtn").on('click', function (e) {
        dz.processQueue();		
    });
}

$(document).ready(function() {
    console.log("ready!");
    $("#error").hide();
    $("#resultHolder").show();
    $("#divClassTable").hide();

    // Smooth scroll for navigation links
    $('a[href^="#"]').on('click', function(e) {
        e.preventDefault();
        var target = this.hash;
        var $target = $(target);
        
        if ($target.length) {
            $('html, body').animate({
                'scrollTop': $target.offset().top - 70 // Offset for fixed header
            }, 800, 'swing');
        }
    });

    // Sample image click handlers - redirect to Google Images
    $('.sample-img').on('click', function() {
        const searchQuery = $(this).data('search');
        if (searchQuery) {
            const googleImagesUrl = `https://www.google.com/search?tbm=isch&q=${encodeURIComponent(searchQuery)}`;
            window.open(googleImagesUrl, '_blank');
        }
    });

    init();
});

function handleClassificationResult(data) {
    if (!data || data.length == 0) {
        $("#resultHolder .empty-state").show();
        $("#divClassTable").hide();
        $("#error").show();
        return;
    }

    let match = null;
    let bestScore = -1;
    for (let i = 0; i < data.length; ++i) {
        let maxScoreForThisClass = Math.max(...data[i].class_probability);
        if (maxScoreForThisClass > bestScore) {
            match = data[i];
            bestScore = maxScoreForThisClass;
        }
    }

    if (match) {
        $("#error").hide();
        $("#resultHolder .empty-state").hide();
        $("#resultHolder").show();
        $("#divClassTable").show();

        let playerClass = match.class;
        let playerName = getPlayerName(playerClass);
        let imageSrc = getPlayerImage(playerClass);

        $("#resultHolder").html(`
            <div class="result-card">
                <img src="${imageSrc}" alt="Player Image" class="result-img">
                <h5 class="result-name">${playerName}</h5>
            </div>
        `);

        let classDictionary = match.class_dictionary;
        for (let personName in classDictionary) {
            let index = classDictionary[personName];
            let probabilityScore = match.class_probability[index];
            let elementName = "#score_" + personName;
            $(elementName).html(probabilityScore.toFixed(2) + "%");
        }
    }
}

function getPlayerName(playerClass) {
    const players = {
        'chritiano_ronaldo': 'Cristiano Ronaldo',
        'gukesh': 'Gukesh Dommaraju',
        'neeraj_chopra': 'Neeraj Chopra',
        'pv_sindhu': 'PV Sindhu',
        'shubhman_gill': 'Shubman Gill'
    };
    return players[playerClass] || 'Unknown Player';
}

function getPlayerImage(playerClass) {
    const images = {
        'chritiano_ronaldo': './IMAGES/ronaldo.jpg',
        'gukesh': './IMAGES/gukesh.jpg',
        'neeraj_chopra': './IMAGES/neeraj.jpg',
        'pv_sindhu': './IMAGES/pv.jpg',
        'shubhman_gill': './IMAGES/gill.jpg'
    };
    return images[playerClass] || '';
}